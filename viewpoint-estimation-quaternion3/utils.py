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


epsilon = 1e-30
nclasses = 2000

def rotate_plane(plane, rotationMatrix):
	rotatedPlane = np.matmul(plane, rotationMatrix)
	return rotatedPlane

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val)/(max_val - min_val)
    img = 255 * img 
    img = 255 - img  # use 255 for background pixel values
    img = np.uint8(img)
    return img

"""
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val)/(max_val - min_val)
    data = 255 * data 
    data = np.uint8(data)
    return data"""

def get_view(img3d, theta):
    #img3d = ndimage.rotate(img3d, theta, axes=(1, 0), reshape=False, mode="constant", 
    #                        cval=np.min(img3d))
    #img3d = tf.keras.preprocessing.image.random_rotation(img3d, theta, row_axis=0, col_axis=1, channel_axis=2, 
    #                   fill_mode='constant', cval=np.min(img3d), interpolation_order=1)
    img3d = tf.keras.preprocessing.image.apply_affine_transform(img3d, theta=theta, tx=0, ty=0, shear=0, zx=1, zy=1, 
                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=np.min(img3d), order=1)
    img = np.sum(img3d, axis=1)
    img = np.transpose(img)
    img = normalize(img)
    return img


def get_view2(img3d, theta, tx, ty):
    #img3d = ndimage.rotate(img3d, theta, axes=(1, 0), reshape=False, mode="constant", 
    #                        cval=np.min(img3d))
    #img3d = tf.keras.preprocessing.image.random_rotation(img3d, theta, row_axis=0, col_axis=1, channel_axis=2, 
    #                   fill_mode='constant', cval=np.min(img3d), interpolation_order=1)
    img3d = tf.keras.preprocessing.image.apply_affine_transform(img3d, theta=theta, tx=0, ty=0, shear=0, zx=1, zy=1, 
                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=np.min(img3d), order=1)
    img = np.sum(img3d, axis=1)
    img = np.transpose(img)
    img = normalize(img)
    img = img[54+tx:454+tx, 63+ty:463+ty]
    return img

def soft_label_encoding(angle, stdev=10):
    angle = int(angle)
    assert angle<360, "viewpoint angle must be less than 360"
    assert stdev%2 == 0, "stdev must be even"
    labels = np.zeros(360)
    central_prob = 0.2
    labels[angle] = central_prob
    for i in range(1, stdev//2):
        labels[(angle-i)%360] = (1-central_prob)/(stdev-2)
        labels[(angle+i)%360] = (1-central_prob)/(stdev-2)		
    return labels

def one_hot_encoding(angle):
    angle = int(angle)
    assert angle<360, "viewpoint angle must be less than 360"
    labels = np.zeros(360)
    labels[angle] = 1.0
    return labels

def rotation_matrix(angle):
    x = angle * math.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    r = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return r

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
    k = 10*tf.constant([i for i in range(nclasses)], dtype=tf.float32)
    gt = 10*gt_class * tf.ones(shape=(nclasses))
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



def get_weights_el(gt_class, sigma=7.0): 
    k = 10*tf.constant([i for i in range(nclasses//2)], dtype=tf.float32) # multiply prediction by bin size to get angle estimate
    gt = 10*gt_class * tf.ones(shape=(nclasses//2))
    distances = angular_distance2(gt, k)
    weights = tf.math.exp(-distances/sigma)
    return weights   

def geom_cross_entropy_el(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1) 
    gt_classes = tf.cast(gt_classes, dtype=tf.float32)
    weights = tf.map_fn(lambda x: get_weights_el(x), gt_classes)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss


# def euler_to_quaternion(r):
#     # print(r)
#     (yaw, pitch, roll) = (r[0], r[1], r[2])
#     qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
#     qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
#     qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
#     qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
#     #print(qx, qy, qz, qw)
#     return [qx, qy, qz, qw]

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

def quaternionAngle(q1, q2):
    q2 = tf.cast(q2, dtype=q1.dtype)
    prod = tf.math.abs(tf.reduce_sum(tf.constant(q1) * tf.constant(q2)))
    if prod > 1.0:
        prod = tf.constant(1.0, dtype=tf.float64)
    theta = tf.math.acos(prod)
    theta = 180.0*theta/np.pi
    #print(theta)
    return theta
