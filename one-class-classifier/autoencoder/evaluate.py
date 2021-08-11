import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
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



# Load ct volume
INPUT_SIZE = (256, 256)
nclasses = 360

# Define the model
with tf.name_scope("Encoder"):
    img_input = tf.keras.layers.Input(shape=(256, 256, 1))
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(img_input)
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(maxpool1)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(maxpool2)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(maxpool3)
    encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

with tf.name_scope("Decoder"):
    deconv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(encoded)
    upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv1)
    deconv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(upsample1)
    upsample2 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv2)
    deconv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(upsample2)
    upsample3 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv3)
    deconv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(upsample3)
    upsample4 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv4)
    decoded = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(upsample4)


encoder = tf.keras.models.Model(img_input, encoded)
autoencoder = tf.keras.models.Model(img_input, decoded)

autoencoder.build(input_shape=(None, 256, 256, 1))
autoencoder.summary()



# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

@tf.function
def test_step(images):
    predictions = autoencoder(images, training=False)
    return predictions


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=autoencoder, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/uncertainty-measure/one-class-classifier/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[0]) 

scales = ["80", "90", "100", "110", "120"]
testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]
scan = testScans[1]

#scales =  ["80", "90", "100", "110", "120"]
decoded_img_list = []
img_list = []
#for scan in testScans:
for view_id in tqdm(range(0, 180, 10), desc="\n{}".format(scan)):
    #for scale in scales:
    
    mae_arr = []
    img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/normals/test2/{}/s100/{}.png".format(scan, view_id), 0)
    #img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/SMIR.Body.025Y.M.CT.57697/{}/{}.png".format(view_id, view_id), 0)
    for ty in range(-20, 21, 5): # Try -20, 21, 2
        mae_list = []
        for tx in range(-20, 21, 5):
            img = img_test[56+ty:456+ty, 56+tx:456+tx]
            img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
            img = np.repeat(img[:,:, np.newaxis], 1, axis=-1)
            img = img/255.0
            x_test = np.expand_dims(img, axis=0)       
            pred = test_step(x_test)
            y_true = tf.reshape(img, [-1])
            y_pred = tf.reshape(pred, [-1])
            mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
            img_list.append(img)
            decoded_img_list.append(pred)
            mae_list.append(mae)
        mae_arr.append(mae_list)
    mae_arr = np.array(mae_arr)

    tx_list = [i for i in range(-20, 21, 5)]
    ty_list = [i for i in range(-20, 21, 5)]

    #sns.set(font_scale=3)
    plt.figure()
    plt.title("Mean absolute error heatmap")
    plt.xlabel("tx (pixels)")
    plt.ylabel("ty (pixels)")
    sns.heatmap(mae_arr, xticklabels=tx_list, yticklabels=ty_list, vmin=0.03, vmax=0.05)
    plt.xlabel("tx (pixels)")
    plt.ylabel("ty (pixels)")
    plt.savefig("hm_mae_{}.png".format(view_id))

# n = 20   
# plt.figure(figsize=(60, 8))        
# for i in range(n):
#     # display original
#     plt.subplot(2, n, i+1)
#     img = np.squeeze(img_list[i])
#     #img = cv.resize(img, dsize=(128, 128), interpolation=cv.INTER_NEAREST)
#     plt.imshow(img)
#     plt.gray()

#     # display feature map
#     # plt.subplot(3, n, i+1 + n)
#     # plt.imshow(np.squeeze(encoded_imgs[i]))
#     # plt.gray()

#     # display reconstruction
#     plt.subplot(2, n, i+1 + n)
#     plt.imshow(np.squeeze(decoded_img_list[i]))
#     plt.gray()
    
# plt.show()
# plt.savefig("encodings_{}.png".format(scan))    
