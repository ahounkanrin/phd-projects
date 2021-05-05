import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import rotation_matrix, one_hot_encoding,  geodesic_distance, angular_distance
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2
import seaborn as sns 

random.seed(0)
min_ctnumber = -1024

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

def rotate_plane(plane, rotationMatrix): 
	return np.matmul(rotationMatrix, plane)

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

# Load ct volume
INPUT_SIZE = (200, 200)

# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)
model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

@tf.function
def test_step(images, labels):
    predictions = model(images)
    return predictions


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/geom-loss-out-of-plane-rotation-ct3/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1]) 

#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/geom-loss-out-of-plane-rotation2/checkpoints/ckpt-40")   
scales = ["80", "90", "100", "110", "120"]

min_errors = []
#err_list = []
#pred_list = []
for view_id in range(0, 360):
	err_list = []
	for scale in scales:
		img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/test-data/test-SMIR.Body.041Y.F.CT.57699/s{}/test{}.png".format(scale, view_id), 0)
		for ty in tqdm(range(1), desc="ty"):
			for tx in tqdm(range(1), desc="tx"):
				img = img_test[56+ty:456+ty, 56+tx:456+tx]
				img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
				img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
				img = img/255.0
				x_test = np.expand_dims(img, axis=0)
				y_test = np.array(one_hot_encoding(view_id))        
				pred = np.argmax(test_step(x_test, y_test)) 
				gt = np.argmax(y_test) 
				#error = geodesic_distance([gt, pred])
				error = angular_distance(gt, pred) 
				#print("Error = {:.4f}".format(error))
				err_list.append(error)
				
	min_errors.append(np.min(err_list))
	    
gt = [i for i in range(360)]
thresholds = [theta for theta in range(0, 95, 5)]

for theta in thresholds:
    acc_bool = np.array([min_errors[i] <= theta  for i in range(len(min_errors))])
    acc = np.mean(acc_bool)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))


