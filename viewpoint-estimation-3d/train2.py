import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import get_view, rotation_matrix, soft_label_encoding, one_hot_encoding
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage




INPUT_SIZE = (200, 200)
imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
imgpath2 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"

train_img3d = nib.load(imgpath1).get_fdata().astype(int)
train_img3d = np.squeeze(train_img3d)
train_img3d = train_img3d[:,:, :512]

val_img3d = nib.load(imgpath2).get_fdata().astype(int)
val_img3d = np.squeeze(val_img3d)
val_img3d = val_img3d[:,:, :512]


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10000000, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

# Define the model

baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
model.summary()
"""
model = myModel(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), nclasses=360)
model.build(input_shape=(None, INPUT_SIZE[0], INPUT_SIZE[1], 3))
model.summary()"""

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
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation 
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

epoch = 0
for step in range(args.steps):
    tic = time.time()
    angles = [random.randint(0, 359) for i in range(args.batch_size)]
    y_train = [soft_label_encoding(i) for i in angles]
    y_train = np.array(y_train)
    x_train = []
    for theta in angles:
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        img = get_view(train_img3d, theta, tx, ty)
        img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
        img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
        #img = tf.keras.preprocessing.image.random_shift(img, wrg=0.1, hrg=0.1, row_axis=0, 
        #      col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0, interpolation_order=1)
        #img = tf.keras.preprocessing.image.random_rotation(img, rg=30, row_axis=0, 
        #      col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0, interpolation_order=1)
        x_train.append(img)
        #plt.imshow(img, cmap='gray')
        #plt.show()
    x_train = np.array(x_train)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)

    # Save logs with TensorBoard Summary
    if step == 0:
        train_logdir = "./logs/train"
        val_logdir = "./logs/val"
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        val_summary_writer = tf.summary.create_file_writer(val_logdir)
        tf.summary.trace_on(graph=True)
    
    for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        train_step(images, labels)
        if step == 0:
            with train_summary_writer.as_default():
                tf.summary.trace_export(name="InceptionV3", step=0)
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=step)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=8)
        toc = time.time()
        print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
            train_loss.result(), train_accuracy.result(), toc-tic))
        # Reset metrics for the next iteration
        train_loss.reset_states()
        train_accuracy.reset_states()

    if (step+1)%200 == 0:
        epoch +=1
        val_views = [i for i in range(360)]
        random.shuffle(val_views)
        y_val = [soft_label_encoding(view) for view in val_views]
        y_val = np.array(y_val)
        x_val = []
        for theta in tqdm(val_views, desc="Generating validation data"):
            img = get_view(val_img3d, theta, 0, 0)
            img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
            x_val.append(img)
        x_val = np.array(x_val)
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

        for test_images, test_labels in tqdm(val_data.map(preprocess, 
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            test_step(test_images, test_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
            tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
            tf.summary.image("val_images", test_images, step=step, max_outputs=8)

        ckpt_path = manager.save()
        template = "\t Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
        print(template.format(test_loss.result(), test_accuracy.result(), ckpt_path))
        
        # Reset metrics for the next epoch
        test_loss.reset_states()
        test_accuracy.reset_states()

