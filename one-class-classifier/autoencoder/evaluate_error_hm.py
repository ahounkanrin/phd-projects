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

H = lambda p: -p*tf.math.log(p)

# Define the model
INPUT_SIZE = (200, 200)
nclasses = 1

# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    return predictions

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/one-class-classifier/cnn/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[11])   
scales =  ["80", "90", "100", "110", "120"]
scan = testScans[0]

for view_id in range(0, 180, 10):
    if not os.path.isdir("view{}".format(view_id)):
            os.mkdir("view{}".format(view_id))
    for scale in scales:
        img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/normals/test2/{}/s{}/{}.png".format(scan, scale, view_id), 0)
        #f = open("errors.txt", "+a")
        confidence_arr = []
        for ty in tqdm(range(-20, 21, 5), desc="ty"):
            confidence_list = []
            for tx in tqdm(range(-20, 21, 5), desc="tx"):
                img = img_test[56+ty:456+ty, 56+tx:456+tx]
                img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
                img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
                img = img/255.0
                x_test = np.expand_dims(img, axis=0)
                y_test = np.array(one_hot_encoding(view_id))
                #preds_vector = test_step(x_test, y_test)
                #pred_entropy = tf.map_fn(H, preds_vector) 
                #entropy = tf.math.reduce_sum(pred_entropy)       
                pred = test_step(x_test, y_test)
                #gt = np.argmax(y_test) 

                #error = geodesic_distance([gt, pred])
                #error = angular_distance(gt, pred) 
                #print("Error = {:.4f}".format(error))
                confidence_list.append(np.squeeze(pred))

            confidence_arr.append(confidence_list)
        confidence_arr = np.array(confidence_arr)
        #f.close()
        tx_list = [i for i in range(-20, 21, 5)]
        ty_list = [i for i in range(-20, 21, 5)]

        #sns.set(font_scale=3)
        plt.figure()
        plt.title("Confidence heatmap - scale: {}%".format(scale))
        plt.xlabel("tx (pixles)")
        plt.ylabel("ty (pixels)")
        sns.heatmap(confidence_arr, xticklabels=tx_list, yticklabels=ty_list, vmin=0.5, vmax=1.0)
        plt.xlabel("tx (pixles)")
        plt.ylabel("ty (pixels)")
        plt.savefig("./view{}/s{}.png".format(view_id, scale))




