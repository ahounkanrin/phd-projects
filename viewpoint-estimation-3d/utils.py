import math
import sys
import numpy as np
from scipy.linalg import logm

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
