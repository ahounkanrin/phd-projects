import os
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["OPENBLAS_NUM_THREADS"] = "8" 
os.environ["MKL_NUM_THREADS"] = "8" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" 
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import math
import sys
from types import new_class
import numpy as np
from scipy.linalg import logm
import tensorflow as tf
from scipy import ndimage
import pandas as pd 

epsilon = 1e-30
nclasses = 500

train_df = pd.read_csv("train2.csv", sep=",")
qw = train_df["qw"].astype(float)
qx = train_df["qx"].astype(float)
qy = train_df["qy"].astype(float)
qz = train_df["qz"].astype(float)
q_class = train_df["class"].astype(int)

vals_qw = tf.constant([qw[i] for i in range(nclasses)], dtype=tf.float32)
vals_qx = tf.constant([qx[i] for i in range(nclasses)], dtype=tf.float32)
vals_qy = tf.constant([qy[i] for i in range(nclasses)], dtype=tf.float32)
vals_qz = tf.constant([qz[i] for i in range(nclasses)], dtype=tf.float32)
keys_tensor = tf.constant([i for i in range(nclasses)])

hashtable_qw = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qw), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qx), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qy), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qz), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)

def rotate_plane(plane, rotationMatrix):
	rotatedPlane = np.matmul(plane, rotationMatrix)
	return rotatedPlane

def geodesic_distance(angles):
    rotmat1 = rotation_matrix(angles[0])
    rotmat2 = rotation_matrix(angles[1])
    rotmat = np.matmul(rotmat1.transpose(), rotmat2)
    norm_frob = np.linalg.norm(logm(rotmat))
    dist = norm_frob / np.sqrt(2.)
    dist_deg = dist * 180./np.pi
    return dist_deg

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

def angular_distance(angle1, angle2):
    dist1 = tf.math.mod(tf.math.abs(angle1 - angle2), 180)
    dist2 = 180 - dist1
    dist = tf.where(dist1>90, x=dist2, y=dist1)
    return dist

def angular_distance2(angle1, angle2):
    dist1 = tf.math.mod(tf.math.abs(angle1 - angle2), 360)
    dist2 = 360 - dist1
    dist = tf.where(dist1>180, x=dist2, y=dist1)
    return dist

def get_weights(gt_class, sigma=7.0): 
    k = tf.constant([i for i in range(nclasses)], dtype=tf.float32)
    gt = gt_class * tf.ones(shape=(nclasses))
    distances = angular_distance2(gt, k)
    weights = tf.math.exp(-distances/sigma)
    return weights

def geom_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    gt_classes = tf.cast(gt_classes, dtype=tf.float32)
    weights = tf.map_fn(lambda x: get_weights(x), gt_classes)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss

def quaternion_get_weights(gt_class, sigma=10): 
    k = tf.constant([i for i in range(nclasses)])
    gt = tf.cast(gt_class, dtype=tf.int32) * tf.ones_like(k)
    k_qw, k_qx, k_qy, k_qz = hashtable_qw.lookup(k), hashtable_qx.lookup(k), hashtable_qy.lookup(k), hashtable_qz.lookup(k)
    k_quaternions = tf.stack([k_qw, k_qx, k_qy, k_qz], axis=-1)
    gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw.lookup(gt), hashtable_qx.lookup(gt), hashtable_qy.lookup(gt), hashtable_qz.lookup(gt)
    gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)
    distances= quaternion_distance(gt_quaternions, k_quaternions)
    weights = tf.math.exp(-sigma * distances)
    return weights

def quaternion_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    weights = tf.map_fn(lambda x: quaternion_get_weights(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss

def euler_to_quaternion(r):
    (pitch, yaw, roll) = (r[0], r[1], r[2])
    qx = np.cos(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.sin(pitch/2) * np.sin(roll/2)
    qy = np.cos(yaw/2) * np.sin(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.cos(pitch/2) * np.sin(roll/2)
    qz = np.sin(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) - np.cos(yaw/2) * np.sin(pitch/2) * np.sin(roll/2)
    qw = np.cos(yaw/2) * np.cos(pitch/2) * np.sin(roll/2) - np.sin(yaw/2) * np.sin(pitch/2) * np.cos(roll/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [pitch, yaw, roll]

def quaternionLoss(predictions, labels):
    assert predictions.shape == labels.shape
    predNorm = tf.broadcast_to(tf.norm(predictions, axis=-1, keepdims=True), shape=predictions.shape)
    predictions = tf.divide(predictions, predNorm)
    labels = tf.cast(labels, dtype=tf.float32)
    loss_batch = 1 - tf.math.square(tf.reduce_sum(predictions * labels, axis=-1))
    #loss = tf.math.reduce_mean(loss_batch)
    return loss_batch #, predictions

def quaternion_distance(q1, q2):
    #q2 = tf.cast(q2, dtype=q1.dtype)
    prod = tf.math.abs(tf.reduce_sum(q1 * q2, axis=-1))
    prod2 = tf.where(prod>1.0, x=tf.ones_like(prod), y=prod)
    dist = 1.0 - prod2
    return dist

def quaternion_angle(q1, q2):
    prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    return theta


if __name__ == "__main__":
    weights = quaternion_get_weights(100)
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(weights.numpy())
    plt.savefig("label_weights100_3.png")
    print("figure saved")
