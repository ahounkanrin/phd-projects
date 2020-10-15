import math
import sys
import numpy as np
from scipy.linalg import logm
import tensorflow as tf

epsilon = 1e-30


def rotation_matrix(angle):
    angle = tf.cast(angle, dtype=tf.float32)
    x = angle * math.pi / 180.
    r11, r12, r13 = tf.math.cos(x), - tf.math.sin(x), tf.constant(0.)
    r21, r22, r23 = tf.math.sin(x), tf.math.cos(x), tf.constant(0.)
    r31, r32, r33 = tf.constant(0.), tf.constant(0.), tf.constant(1.)
    row1 = tf.stack([r11, r12, r13])
    row2 = tf.stack([r21, r22, r23])
    row3 = tf.stack([r31, r32, r33])
    r = tf.stack([row1, row2, row3])
    return r

def geodesic_distance(angles):
    rotmat1 = rotation_matrix(angles[0])
    rotmat2 = rotation_matrix(angles[1])
    rotmat = tf.linalg.matmul(tf.transpose(rotmat1), rotmat2)
    rotmat = tf.cast(rotmat, dtype=tf.complex64)
    log_rotmat =  tf.linalg.logm(rotmat)
    log_rotmat = tf.cast(log_rotmat, dtype=tf.float32)
    norm_frob = tf.linalg.norm(log_rotmat)
    dist = norm_frob / np.sqrt(2.)
    dist_deg = dist * 180./np.pi
    return dist_deg

def angular_distance(angle1, angle2):
    dist1 = tf.math.mod(tf.math.abs(angle1 - angle2), 360)
    dist2 = 360 - dist1
    dist = tf.where(dist1>180, x=dist2, y=dist1)
    return dist

def get_weights(gt_class, sigma=7): # sigma changed from 3 to 1
    k = tf.constant([i for i in range(360)], dtype=tf.float32)
    gt = gt_class * tf.ones(shape=(360))
    distances = angular_distance(gt, k)
    weights = tf.math.exp(-distances/sigma)
    return weights

def geom_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    gt_classes = tf.cast(gt_classes, dtype=tf.float32)
    weights = tf.map_fn(lambda x: get_weights(x), gt_classes)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss

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
