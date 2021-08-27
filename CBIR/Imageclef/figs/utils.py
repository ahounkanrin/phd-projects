import math
import sys
import numpy as np
from scipy.linalg import logm
import tensorflow as tf
from scipy import ndimage
from matplotlib import pyplot as plt
import pandas as pd

epsilon = 1e-30

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


def get_weights(gt_class, sigma=5.0): 
    k = tf.constant([i for i in range(193)], dtype=tf.float32)
    gt = gt_class * tf.ones(shape=(193))
    distances = tf.math.abs(gt - k)
    weights = tf.math.exp(-distances/sigma)
    return weights

def geom_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    gt_classes = tf.cast(gt_classes, dtype=tf.float32)
    weights = tf.map_fn(lambda x: get_weights(x), gt_classes)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss


train_df = pd.read_csv("train_old.csv", sep=";")
test_df = pd.read_csv("test_old.csv", sep=";")
classes = sorted(list(set(train_df["irma_code"])))
decod_dict = dict((x, y) for x,y in enumerate(classes))
def irma_distance(code1, code2):
    code1_axis_T, code1_axis_D, code1_axis_A, code1_axis_B = code1.split("-")
    code2_axis_T, code2_axis_D, code2_axis_A, code2_axis_B = code2.split("-")
    error_T = 0.
    for i in range(4):
        error_T += 1./(i+1) * int(code1_axis_T[i] != code2_axis_T[i]) # int(code1_axis_T[i] != code2_axis_T[i]) is an indicator function that outputs 1 if characters don't match and 0 otherwise   
    error_T = error_T * 0.25 / (1 + 1/2 + 1/3 + 1/4)

    error_D = 0.
    for i in range(3):
        error_D += 1./(i+1) * int(code1_axis_D[i] != code2_axis_D[i])   
    error_D = error_D * 0.25 / (1 + 1/2 + 1/3)

    error_A = 0.
    for i in range(3):
        error_A += 1./(i+1) * int(code1_axis_A[i] != code2_axis_A[i])   
    error_A = error_A * 0.25 / (1 + 1/2 + 1/3)

    error_B = 0.
    for i in range(3):
        error_B += 1./(i+1) * int(code1_axis_B[i] != code2_axis_B[i])   
    error_B = error_B * 0.25 / (1 + 1/2 + 1/3)
    error = error_A + error_B + error_D + error_T
    return error

def irma_get_weights(gt_class, sigma=20): 
    k = [i for i in range(193)]
    gt = [gt_class for i in range(193)]
    k_codes = [decod_dict[i] for i in k]
    gt_codes = [decod_dict[i] for i in gt]
    distances= np.array([irma_distance(k_codes[i], gt_codes[i]) for i in range(193)])
    weights = np.exp(-sigma * distances)
    return weights

def irma_cross_entropy(predictions, labels):
    gt_classes = tf.argmax(labels, axis=-1)
    #gt_classes = tf.cast(gt_classes, dtype=tf.float32)
    weights = tf.map_fn(lambda x: irma_get_weights(x), gt_classes, fn_output_signature=tf.float32)
    pred_log = tf.math.log(predictions + epsilon)
    loss = - weights * pred_log
    return loss

if __name__ == "__main__":
    weights = irma_get_weights(150)
    plt.plot(np.arange(193), weights, ".")
    plt.xlabel("Classes")
    plt.ylabel("Weights")
    plt.savefig("irma_class_weights_150.png")
    #print(weights)

    

