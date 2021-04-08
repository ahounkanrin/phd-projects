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
# imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"
# N = 512 
# print("INFO: loading CT volume...")
# tic_load = time.time()
# test_ctVolume = nib.load(imgpath).get_fdata().astype(int)
# test_ctVolume = np.squeeze(test_ctVolume)
# test_voi = test_ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
# #test_voi = normalize(test_voi)    # Rescale CT numbers between 0 and 255
# test_voi = test_voi - min_ctnumber
# test_voi = np.pad(test_voi, N//2, "constant", constant_values=0)
# toc_load = time.time()
# print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

# tic_fft = time.time()
# test_voiShifted = np.fft.fftshift(test_voi)
# test_voiFFT = np.fft.fftn(test_voiShifted)
# test_voiFFTShifted = np.fft.fftshift(test_voiFFT)
# toc_fft = time.time()
# print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


# # Rotation and Interpolation of the projection slice from the 3D FFT volume
# x_axis = np.linspace(-N+0.5, N-0.5, 2*N)
# y_axis = np.linspace(-N+0.5, N-0.5, 2*N)
# z_axis = np.linspace(-N+0.5, N-0.5, 2*N)

# projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
# projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

# def render_test_view(viewpoint):
#     theta = viewpoint[0]
#     tx = viewpoint[1]
#     ty = viewpoint[2]
#     rotationMatrix = rotation_matrix(theta)
#     projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
#     projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=test_voiFFTShifted, xi=projectionSlice, method="linear",
#                                     bounds_error=False)      
#     img = np.abs(fftshift(ifft2(projectionSliceFFT)))
#     img = img[N//2:N+N//2, N//2:N+N//2]
#     img = normalize(img)
#     img = img[56+ty:456+ty, 56+tx:456+tx]
#     img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
#     img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
#     label = one_hot_encoding(theta)
#     return img, label


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
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/geom-loss-out-of-plane-rotation2/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

#checkpoint.restore(manager.checkpoints[-1]) 

checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/geom-loss-out-of-plane-rotation2/checkpoints/ckpt-40")   
scales = ["80", "90", "100", "110", "120"]

for view_id in range(0, 180, 5):

    for scale in scales: 	
        if not os.path.isdir("view{}".format(view_id)):
            os.mkdir("view{}".format(view_id))
        img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/test/s{}/test{}.png".format(scale, view_id), 0)
        #f = open("errors.txt", "+a")
        err_arr = []
        for ty in tqdm(range(-20, 21, 2), desc="ty"):
            err_list = []
            for tx in tqdm(range(-20, 21, 2), desc="tx"):
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

            err_arr.append(err_list)
        err_arr = np.array(err_arr)
        #f.close()
        tx_list = [i for i in range(-20, 21, 2)]
        ty_list = [i for i in range(-20, 21, 2)]

        #sns.set(font_scale=3)
        plt.figure()
        plt.title("scale: {}%".format(scale))
        plt.xlabel("tx (pixles)")
        plt.ylabel("ty (pixels)")
        sns.heatmap(err_arr, xticklabels=tx_list, yticklabels=ty_list, vmin=0.0, vmax=90.0)
        plt.xlabel("tx (pixles)")
        plt.ylabel("ty (pixels)")
        plt.savefig("./view{}/s{}.png".format(view_id, scale))

    


