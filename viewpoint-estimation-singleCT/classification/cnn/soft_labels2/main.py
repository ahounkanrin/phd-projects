import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
from utils import rotation_matrix, geodesic_distance

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

args = get_arguments()

def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')
    x_val = hf.get("x_val")
    y_val = hf.get("y_val")
    x_test = hf.get('x_test')
    y_test = hf.get('y_test')

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

# Load dataset
print("INFO: Loading dataset...")
INPUT_SIZE = (200, 200)
DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"
x_train, y_train, x_val, y_val, x_test, y_test = read_dataset(DIR+'chest_fov_400x400_soft_labels.h5')

xtrain = []
xval = []
xtest = []
for i in range(len(x_train)):
    xtrain.append(cv.resize(x_train[i], INPUT_SIZE, interpolation = cv.INTER_AREA))
for i in range(len(x_val)):
    xval.append(cv.resize(x_val[i], INPUT_SIZE, interpolation = cv.INTER_AREA)) 
for i in range(len(x_test)):
    xtest.append(cv.resize(x_test[i], INPUT_SIZE, interpolation = cv.INTER_AREA))

x_train = np.array(xtrain)
x_val = np.array(xval)
x_test = np.array(xtest)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size) 
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
print("[INFO] Datasets loaded...")

# Define the model

baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
#model.build((None, 100, 100, 1))
model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
test_loss = tf.keras.metrics.CategoricalCrossentropy(name="test_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation purposes
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Calculate metrics
    train_accuracy.update_state(labels, predictions)
    train_loss.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    test_loss.update_state(labels, predictions)
    test_accuracy.update_state(labels, predictions)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "./checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

if args.is_training:

    # Save logs with TensorBoard Summary
    train_logdir = "./logs/train"
    val_logdir = "./logs/val"
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)

    with train_summary_writer.as_default():
        tf.summary.histogram(name="train_classes", data=np.argmax(y_train), step=0)
    with val_summary_writer.as_default():
        tf.summary.histogram(name="val_classes", data=np.argmax(y_val), step=0)
    tf.summary.trace_on(graph=True)    

    # Training loop
    step = 0
    for epoch in tqdm(range(args.epochs)):
        for images, labels in tqdm(train_data.map(preprocess, 
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Training"):
            train_step(images, labels)
            if step == 0:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="InceptionV3", step=0)
            step += 1
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=step)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                tf.summary.image("image", images, step=step, max_outputs=8)
            print("Step {}: \t loss = {:.4f} \t acc = {:.4f}".format(step, train_loss.result(), train_accuracy.result()))
            # Reset metrics for the next iteration
            train_loss.reset_states()
            train_accuracy.reset_states()

        for test_images, test_labels in tqdm(val_data.map(preprocess, 
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            test_step(test_images, test_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
            tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
            tf.summary.image("val_images", test_images, step=epoch, max_outputs=16)

        ckpt_path = manager.save()
        template = "\t Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
        print(template.format(epoch+1, test_loss.result(), test_accuracy.result(), ckpt_path))
        
        # Reset metrics for the next epoch
        test_loss.reset_states()
        test_accuracy.reset_states()
else:
    checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation/classification/cnn/soft_labels2/checkpoints_200x200/ckpt-100") 
    #checkpoint.restore(manager.checkpoints[-1]) 
    pred = []
    for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            test_step(test_images, test_labels)
            pred.append(np.argmax(model(test_images)))
            
    gt = [np.argmax(label) for label in y_test]
    thresholds = [theta for theta in range(0, 60, 5)]

    error1 = np.abs(np.array(pred) - np.array(gt)) % 360         
    error2 = [geodesic_distance(rotation_matrix(gt[i]), rotation_matrix(pred[i])) for i in range(len(gt))]
    
    print("\n\nMedian Error = {:.4f}".format(np.median(np.array(error2))))
    with open("soft_classification_300x300.txt", "w") as f:
        print("Median Error = {:.4f}".format(np.median(np.array(error2))), file=f)

    acc_list2 = []

    for theta in thresholds:
        acc_bool2 = np.array([error2[i] <= theta  for i in range(len(error2))])
        acc2 = np.mean(acc_bool2)
        acc_list2.append(acc2)
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2))
        with open("soft_classification_300x300.txt", "a") as f:
            print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2), file=f)
        
