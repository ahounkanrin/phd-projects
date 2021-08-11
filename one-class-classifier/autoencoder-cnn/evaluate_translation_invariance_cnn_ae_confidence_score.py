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
INPUT_SIZE = (200, 200)
nclasses = 360

# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()


# Define the model
# inputs2 = tf.keras.Input(shape=(256, 256, 1))
# #x = baseModel(inputs)
# x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
# x2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2) 
# x2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(x2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
# x2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same")(x2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
# x2 = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), activation="relu", padding="same")(x2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
# x2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(x2)
# x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
# x2 = tf.keras.layers.Flatten()(x2)
# #x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x2 = tf.keras.layers.Dense(2048, activation="relu")(x2)
# x2 = tf.keras.layers.Dropout(rate=0.2)(x2)
# x2 = tf.keras.layers.Dense(1024, activation="relu")(x2)
# x2 = tf.keras.layers.Dropout(rate=0.2)(x2)
# outputs2 = tf.keras.layers.Dense(1, activation="sigmoid")(x2)
# model2 = tf.keras.Model(inputs=inputs2, outputs=outputs2)

# model2.summary()


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

#autoencoder.build(input_shape=(None, 256, 256, 1))

with tf.name_scope("oc_cnn"):
    #baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
    inputs = tf.keras.Input(shape=(256, 256, 1))
    #x = baseModel(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x) 
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model_cnn = tf.keras.Model(inputs=inputs, outputs=outputs)




# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

checkpoint_ae = tf.train.Checkpoint(model=autoencoder, optimizer=optimizer)
checkpoint_dir_ae = "/scratch/hnkmah001/phd-projects/one-class-classifier/autoencoder/checkpoints/"
manager_ae = tf.train.CheckpointManager(checkpoint_ae, directory=checkpoint_dir_ae, max_to_keep=10)
checkpoint_ae.restore(manager_ae.checkpoints[-1])
autoencoder.summary()

checkpoint_cnn = tf.train.Checkpoint(model=model_cnn, optimizer=optimizer)
checkpoint_dir_cnn = "/scratch/hnkmah001/phd-projects/one-class-classifier/autoencoder-cnn/checkpoints/"
manager_cnn = tf.train.CheckpointManager(checkpoint_cnn, directory=checkpoint_dir_cnn, max_to_keep=50)
checkpoint_cnn.restore(manager_cnn.checkpoints[-1])
model_cnn.summary()

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/exp1-no-aug/checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)
checkpoint.restore(manager.checkpoints[11]) 
model.summary()

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    return predictions

#Define checkpoint manager to save model weights

scales = ["80", "90", "100", "110", "120"]
testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]

scan = testScans[7]
min_errors = []

#for scan in testScans:
for view_id in tqdm(range(0, 360), desc="\n{}".format(scan)):
    confidence_list = []
    pred_list = []
    
    y_test = np.array(one_hot_encoding(view_id)) 
    gt = np.argmax(y_test)
    img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/normals/test2/{}/s100/{}.png".format(scan, view_id), 0)
    #img_test = cv.imread("/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/SMIR.Body.025Y.M.CT.57697/{}/{}.png".format(view_id, view_id), 0)
    for ty in range(-20, 21, 5): 
        for tx in range(-20, 21, 5):
            img0 = img_test[56+ty:456+ty, 56+tx:456+tx]
            img = cv.resize(img0, INPUT_SIZE, interpolation=cv.INTER_AREA)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
            img = img/255.0
            x_test = np.expand_dims(img, axis=0)
            predictions =  model(x_test, training=False) 
            pred = np.argmax(predictions)

            img2 = cv.resize(img0, (256, 256), interpolation=cv.INTER_AREA)
            img2 = np.repeat(img2[:,:, np.newaxis], 1, axis=-1)
            img2 = img2/255.0
            x_test2 = np.expand_dims(img2, axis=0)
            x_test2 = autoencoder(x_test2)
            confidence = model_cnn(x_test2, training=False)      
             
            confidence_list.append(confidence)
            pred_list.append(pred)

    error = angular_distance(gt, pred_list[np.argmax(confidence_list)])        
    min_errors.append(error)
           
thresholds = [theta for theta in range(0, 95, 5)]
acc_list = []
for theta in thresholds:
    acc_bool = np.array([min_errors[i] <= theta  for i in range(len(min_errors))])
    acc = np.mean(acc_bool)
    acc_list.append(acc)
    # print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))

print("acc = ",  acc_list)
f = open("acc_{}.txt".format(scan), "w+")
print("acc = ",  acc_list, file=f)
f.close()