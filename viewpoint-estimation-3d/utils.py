import math
import sys
import numpy as np
from scipy.linalg import logm
import tensorflow as tf
from scipy import ndimage

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

def geodesic_distance(rotmat1, rotmat2):
    rotmat = np.matmul(rotmat1.transpose(), rotmat2)
    norm_frob = np.linalg.norm(logm(rotmat))
    dist = norm_frob / np.sqrt(2.)
    dist_deg = dist * 180./np.pi
    return dist_deg

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val)/(max_val - min_val)
    data = 255 * data 
    data = np.uint8(data)
    return data

def get_view(img3d, theta, tx, ty):

    img3d = ndimage.rotate(img3d, theta, axes=(1, 0), reshape=False, mode="constant", 
                            cval=np.min(img3d))
    img = np.sum(img3d, axis=1)
    img = np.transpose(img)
    img = normalize(img)
    img = img[54+ty:454+ty, 63+tx:463+tx]
    return img
    

