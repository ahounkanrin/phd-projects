import math
import numpy as np
from scipy.linalg import logm


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


def dist_abs(pred_angle, gt_real):
    print("gt_real", gt_real.shape)
    batch_size = gt_real.shape[0]
    error_azi = np.abs([(gt_real[i, 0] - pred_angle[i][0]) % 360 for i in range(batch_size)])
    error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    return error_azi
