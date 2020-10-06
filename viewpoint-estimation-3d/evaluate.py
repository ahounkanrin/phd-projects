#import nibabel as nib
import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import rotation_matrix, soft_label_encoding, one_hot_encoding, geodesic_distance
from tqdm import tqdm
import time

eng = matlab.engine.start_matlab()

INPUT_SIZE = (256, 256)
#DATA_DIR = "/home/anicet/Datasets/ctfullbody/"
imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
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
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1]) 

y_test = [soft_label_encoding(i) for i in range(360)]  
y_test = np.array(y_test)
x_test = []
for theta in tqdm(range(360), desc="Generating test data"):  
    img = eng.projection2d(imgpath, theta, 0, 0, "z")
    img = np.array(img).astype("uint8")
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.stack((img, img, img), axis=-1)
    x_test.append(img)
x_test = np.array(x_test)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

pred = []
for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Evaluation"):
        test_step(test_images, test_labels)
        pred.append(np.argmax(model(test_images)))
        
gt = [np.argmax(label) for label in y_test]
thresholds = [theta for theta in range(0, 60, 5)]

error1 = np.abs(np.array(pred) - np.array(gt)) % 360         
error2 = [geodesic_distance(rotation_matrix(gt[i]), rotation_matrix(pred[i])) for i in range(len(gt))]

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(error2))))
with open("classification_3d.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(error2))), file=f)

acc_list2 = []

for theta in thresholds:
    acc_bool2 = np.array([error2[i] <= theta  for i in range(len(error2))])
    acc2 = np.mean(acc_bool2)
    acc_list2.append(acc2)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2))
    with open("classification_3d.txt", "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2), file=f)
        


