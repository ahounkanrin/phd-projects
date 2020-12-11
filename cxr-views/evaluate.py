import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import rotation_matrix, one_hot_encoding,  geodesic_distance
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2

random.seed(0)
DATA_DIR = "/scratch/hnkmah001/Datasets/CXR-IndianaUniversity/"
INPUT_SIZE = (200, 200)
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

df = pd.read_csv(DATA_DIR + "indiana_projections.csv")
filenames = df["filename"].tolist()
projections = df["projection"].tolist()
labels = [int(view == "Lateral") for view in projections]
y_test = [one_hot_encoding(i) for i in labels]
x_test = []
for i in tqdm(range(len(filenames)), desc="Reading images"):
    img = cv.imread(DATA_DIR+"images/images_normalized/"+filenames[i])
    img = cv.resize(img,  INPUT_SIZE, interpolation=cv.INTER_AREA)
    x_test.append(img)
x_test = np.array(x_test)
y_test = np.array(y_test)        
print("INFO: shape of x_test", x_test.shape)
print("INFO: shape of y_test", y_test.shape)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)


# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

@tf.function
def test_step(images, labels):
    predictions = model(images)
    test_accuracy.update_state(labels, predictions)
    


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/cxr-views/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1]) 
#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/ckpt-2")    


for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):
    test_step(test_images, test_labels)

print("Accuracy = {:.4f}".format(test_accuracy.result()))
