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
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    #parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    #parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

args = get_arguments()


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

#@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    #labels = tf.one_hot(labels, 360)
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = geom_cross_entropy(predictions, labels)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=strategy.num_replicas_in_sync*images.shape[0])
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

#@tf.function
def test_step(images, labels):
    #labels = tf.one_hot(labels, 360)
    predictions = model(images)
    test_loss = geom_cross_entropy(predictions, labels)
    test_loss = tf.reduce_sum(test_loss, axis=-1)  
    test_loss = tf.nn.compute_average_loss(test_loss, global_batch_size=strategy.num_replicas_in_sync*images.shape[0])
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
    print("INFO: Processing dataset...")
    INPUT_SIZE = (200, 200)
    data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/train-val/"

    train_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=INPUT_SIZE,
    batch_size=args.batch_size,
    label_mode="categorical")
    train_data = train_data.map(lambda x,y: (preprocess(x,y)), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    #train_data = train_data.prefetch(102400)
    train_data = strategy.experimental_distribute_dataset(train_data)

    val_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=INPUT_SIZE,
    batch_size=args.batch_size,
    label_mode="categorical")
    val_data = val_data.map(lambda x,y: preprocess(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_data = val_data.prefetch(102400)
    val_data = strategy.experimental_distribute_dataset(val_data)

    print("Done.")

    baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    data_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.02, 0.02), fill_mode='constant', 
                interpolation='bilinear', seed=123, name=None),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3), 
                fill_mode='constant', interpolation='bilinear', seed=None, name=None)
                ], name="data_augmentation_layer")

    x = data_aug(inputs)
    x = baseModel(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Define cost function, optimizer and metrics
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=500, 
                                                                decay_rate=0.96, staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


    # Define checkpoint manager to save model weights
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/exp3-in-plane-rotation-scaling/checkpoints/"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)


    # Save logs with TensorBoard Summary
    train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/exp3-in-plane-rotation-scaling/logs/train"
    val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/exp3-in-plane-rotation-scaling/logs/val"
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)

    #tf.summary.trace_on(graph=True)    

    # Training loop
    step = 0
    for epoch in tqdm(range(args.epochs)):
        for images, labels in train_data:

            tic = time.time()
            train_loss = distributed_train_step(images, labels)
            
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

