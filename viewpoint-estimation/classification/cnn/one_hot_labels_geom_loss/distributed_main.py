import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
from utils import geodesic_distance, geom_cross_entropy, one_hot_encoding

strategy = tf.distribute.MirroredStrategy()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

args = get_arguments()

def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = np.array(hf.get('x_train'))
    y_train = np.array(hf.get('y_train'))
    x_val = np.array(hf.get("x_val"))
    y_val = np.array(hf.get("y_val"))
    x_test = np.array(hf.get('x_test'))
    y_test = np.array(hf.get('y_test'))
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y



def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = geom_cross_entropy(predictions, labels)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.ngpus*images.shape[0])
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss


def test_step(images, labels):
    predictions = model(images)
    test_loss = geom_cross_entropy(predictions, labels)
    test_loss = tf.reduce_sum(test_loss, axis=-1)  
    test_loss = tf.nn.compute_average_loss(test_loss, global_batch_size=args.ngpus*images.shape[0])
    test_accuracy.update_state(labels, predictions)
    return test_loss

@tf.function
def distributed_train_step(images, labels):
    per_replica_losses = strategy.experimental_run_v2(train_step, args=(images, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_test_step(images, labels):
    per_replica_losses = strategy.experimental_run_v2(test_step, args=(images, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


with strategy.scope():

    # Load dataset
    print("INFO: Loading dataset...")
    INPUT_SIZE = (200, 200)
    DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"
    x_train, y_train, x_val, y_val, x_test, y_test = read_dataset(DIR+'chest_fov_400x400_sparse_labels.h5')

    xtrain = []
    xval = []
    xtest = []
    for i in range(len(x_train)):
        xtrain.append(cv.resize(x_train[i], INPUT_SIZE, interpolation = cv.INTER_AREA))
    for i in range(len(x_val)):
        xval.append(cv.resize(x_val[i], INPUT_SIZE, interpolation = cv.INTER_AREA)) 
    for i in range(len(x_test)):
        xtest.append(cv.resize(x_test[i], INPUT_SIZE, interpolation = cv.INTER_AREA))

    y_train = np.array([one_hot_encoding(i-1) for i in y_train]).astype("float32")
    y_val = np.array([one_hot_encoding(i-1) for i in y_val]).astype("float32")
    y_test = np.array([one_hot_encoding(i-1) for i in y_test]).astype("float32")

    x_train = np.array(xtrain)
    x_val = np.array(xval)
    x_test = np.array(xtest)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size*args.ngpus)
    train_data = train_data.map(lambda x,y: (preprocess(x,y)), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    train_data = strategy.experimental_distribute_dataset(train_data)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size*args.ngpus)
    val_data = val_data.map(lambda x,y: preprocess(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    val_data = strategy.experimental_distribute_dataset(val_data)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
    print("INFO: Datasets loaded.")

    baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
    x = baseModel.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
    #model.summary()

    # Define cost function, optimizer and metrics
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                                decay_rate=0.96, staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


    # Define checkpoint manager to save model weights
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation/classification/cnn/one_hot_labels_geom_loss/checkpoints/"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)

    if args.is_training:

        # Save logs with TensorBoard Summary
        train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation/classification/cnn/one_hot_labels_geom_loss/logs/train"
        val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation/classification/cnn/one_hot_labels_geom_loss/logs/val"
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        val_summary_writer = tf.summary.create_file_writer(val_logdir)

        tf.summary.trace_on(graph=True)    

        # Training loop
        step = 0
        for epoch in tqdm(range(args.epochs)):
            for images, labels in train_data:
                tic = time.time()
                train_loss = distributed_train_step(images, labels)
                
                if step == 0:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export(name="InceptionV3", step=0)
                step += 1
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=step)
                    tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                    #tf.summary.image("image", images, step=step, max_outputs=2)
                toc = time.time()
                print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                        train_loss, train_accuracy.result(), toc-tic))
                train_accuracy.reset_states()

            test_it = 0
            test_loss = 0.
            for test_images, test_labels in tqdm(val_data, desc="Validation"):
                test_loss += distributed_test_step(test_images, test_labels)
                test_it +=1
            test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
            with val_summary_writer.as_default():
                tf.summary.scalar("val_loss", test_loss, step=epoch)
                tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
                #tf.summary.image("val_images", test_images, step=epoch, max_outputs=2)

            ckpt_path = manager.save()
            template = "Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
            print(template.format(epoch+1, test_loss, test_accuracy.result(), ckpt_path))
            
            # Reset metrics for the next epoch
            test_accuracy.reset_states()
    else:
        checkpoint.restore(manager.checkpoints[-1]) 
        pred = []
        for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
                pred.append(np.argmax(model(test_images)))
                
        gt = [np.argmax(label) for label in y_test]
        thresholds = [theta for theta in range(0, 60, 5)]

        error1 = np.abs(np.array(pred) - np.array(gt)) % 360         
        error2 = [geodesic_distance([gt[i], pred[i]]) for i in range(len(gt))]
        
        print("\n\nMedian Error = {:.4f}".format(np.median(np.array(error2))))
        with open("classification_geom_loss.txt", "w") as f:
            print("Median Error = {:.4f}".format(np.median(np.array(error2))), file=f)

        acc_list2 = []

        for theta in thresholds:
            acc_bool2 = np.array([error2[i] <= theta  for i in range(len(error2))])
            acc2 = np.mean(acc_bool2)
            acc_list2.append(acc2)
            print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2))
            with open("classification_geom_loss.txt", "a") as f:
                print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2), file=f)
            
