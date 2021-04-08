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
imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"
N = 512 
print("INFO: loading CT volume...")
tic_load = time.time()
test_ctVolume = nib.load(imgpath).get_fdata().astype(int)
test_ctVolume = np.squeeze(test_ctVolume)
test_voi = test_ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
#test_voi = normalize(test_voi)    # Rescale CT numbers between 0 and 255
test_voi = test_voi - min_ctnumber
test_voi = np.pad(test_voi, N//2, "constant", constant_values=0)
toc_load = time.time()
print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

tic_fft = time.time()
test_voiShifted = np.fft.fftshift(test_voi)
test_voiFFT = np.fft.fftn(test_voiShifted)
test_voiFFTShifted = np.fft.fftshift(test_voiFFT)
toc_fft = time.time()
print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


# Rotation and Interpolation of the projection slice from the 3D FFT volume
x_axis = np.linspace(-N+0.5, N-0.5, 2*N)
y_axis = np.linspace(-N+0.5, N-0.5, 2*N)
z_axis = np.linspace(-N+0.5, N-0.5, 2*N)

projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

def render_test_view(viewpoint):
    theta = viewpoint[0]
    #tx = viewpoint[1]
    #ty = viewpoint[2]
    rotationMatrix = rotation_matrix(theta)
    projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=test_voiFFTShifted, xi=projectionSlice, method="linear",
                                    bounds_error=False)      
    img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    img = img[N//2:N+N//2, N//2:N+N//2]
    img = normalize(img)
    #img = img[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = one_hot_encoding(theta)
    return img, label

xtest = [(theta, 0, 0) for theta in range(-20, 21)]

# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(41, activation="softmax")(x)
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
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/refinement2/viewpoint0_20/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1])  
#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/ckpt-2")    
xtest_epoch = xtest.copy()
x_test = []
y_test = []
while len(xtest_epoch) > 0:
    if len(xtest_epoch) >= cpu_count():
        test_viewpoints_batch = xtest_epoch[:cpu_count()]
    else:
        test_viewpoints_batch = xtest_epoch.copy()
    
    with Pool() as pool:
        test_batch = pool.map(render_test_view, test_viewpoints_batch)
    test_batch = np.array(test_batch)
    for i in range(len(test_batch)):
        x_test.append(test_batch[i, 0])
        y_test.append(test_batch[i, 1])
    
    for example in test_viewpoints_batch:
        xtest_epoch.remove(example)

x_test = np.array(x_test)
y_test = np.array(y_test)        
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

pred = []
for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):
    pred_it = test_step(test_images, test_labels)
    pred.append(np.argmax(pred_it))

gt = [np.argmax(label) for label in y_test]
errors = [geodesic_distance([gt[i], pred[i]]) for i in range(len(gt))]
thresholds = [theta for theta in range(0, 21, 1)]

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
with open("mederr.txt", "w") as f:
	print("Median Error = {:.4f}".format(np.median(np.array(errors))), file=f)

acc_list = []

for theta in thresholds:
	acc_bool = np.array([errors[i] <= theta  for i in range(len(errors))])
	acc = np.mean(acc_bool)
	acc_list.append(acc)
	print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))

df = pd.DataFrame()
df["theta"] = thresholds
df["accuracy"] = acc_list
df.to_csv("accuracy.csv", sep=",", index=False)
