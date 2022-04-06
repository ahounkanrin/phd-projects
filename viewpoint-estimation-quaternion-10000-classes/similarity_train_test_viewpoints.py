import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv

print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)
nclasses = 20000

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

def quaternion_angle(q1, q2):
    prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    return theta

translation_window = [i for i in range(-20, 21, 5)]

def load_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_png(raw_img, channels=3)
    return img 

def crop_image(img):
    tx = 0 #np.random.choice(translation_window)
    ty = 0 #np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(imgpath, labels):
    img = tf.map_fn(load_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.map_fn(crop_image, img)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32))
    return img, labels


# Load dataset
data_dir_test = "/scratch/hnkmah001/Datasets/ctfullbody/quaternions_20000_classes2/test/"
test_df = pd.read_csv("test2.csv", sep=",")
img_path = test_df["image"].apply(lambda imgID: data_dir_test + imgID)
qw = test_df["qw"].astype(float)[:nclasses]
qx = test_df["qx"].astype(float)[:nclasses]
qy = test_df["qy"].astype(float)[:nclasses]
qz = test_df["qz"].astype(float)[:nclasses]
# = test_df["class"].astype(int)
test_quaternions = tf.stack([qw, qx, qy, qz], axis=-1)
test_quaternions = tf.cast(test_quaternions, dtype=tf.float32)

data_dir_train = "/scratch/hnkmah001/Datasets/ctfullbody/quaternions_20000_classes/test/"
train_df = pd.read_csv("train1.csv", sep=",")
qw2 = train_df["qw"].astype(float)[:nclasses]
qx2 = train_df["qx"].astype(float)[:nclasses]
qy2 = train_df["qy"].astype(float)[:nclasses]
qz2 = train_df["qz"].astype(float)[:nclasses]
train_quaternions = tf.stack([qw2, qx2, qy2, qz2], axis=-1)
train_quaternions = tf.cast(train_quaternions, dtype=tf.float32)

viewpoint_distances = []
n = 10000
for i in tqdm(range(nclasses)):
        viewpoint_distances.append(quaternion_angle(train_quaternions[i], test_quaternions[n]))


print(np.argmin(viewpoint_distances))
print("min angle", np.min(viewpoint_distances))
train_img = cv.imread(data_dir_train + "SMIR.Body.025Y.M.CT.57697_{}.png".format(np.argmin(viewpoint_distances)))
test_img = cv.imread(data_dir_test + "SMIR.Body.025Y.M.CT.57697_{}.png".format(n))
cv.imwrite("10000_train.png", train_img)
cv.imwrite("10000_test.png", test_img)

print(sorted(viewpoint_distances)[100])