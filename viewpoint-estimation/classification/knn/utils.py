import math
import sys
import numpy as np
from scipy.linalg import logm


def get_adjusted_pred_uniform_mulaw(pred_angle, gt_real, mu=20, in_range=45, out_range=180, rand_ele=True):
    # from a 180 class to 45 conversion.
    diff = np.zeros(pred_angle.shape)
    diff[:, 0] = mu_inv(np.random.uniform(-in_range, in_range, size=diff.shape[0]))

    if rand_ele:
        diff[:, 1] = np.random.uniform(-5, 25, size=(gt_real.shape[0]))
        diff[:, 2] = np.random.uniform(-10, 10, size=(gt_real.shape[0]))

    pred_angle = np.copy((gt_real + diff) % 360)

    diff = get_actual_diff(gt_real, pred_angle)

    return pred_angle, diff


def mu_inv(inp, mu=20.0, in_range=45.0, out_range=180.0):
    inp = inp / in_range
    out = np.sign(inp) / mu * (np.power(1 + mu, np.abs(inp)) - 1) * out_range
    return out


def mu_law(inp, mu=20.0, in_range=180.0, out_range=45.0):
    inp = inp / in_range
    out = np.sign(inp) * (np.log(1 + mu * np.abs(inp))) / np.log(1 + mu) * out_range
    return out



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




def dist_abs(pred_angle, gt_real, angle_type='azimuth'):
    print("gt_real", gt_real.shape)
    batch_size = gt_real.shape[0]
    if angle_type == 'azimuth':
        error_azi = np.abs([(gt_real[i, 0] - pred_angle[i][0]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    elif angle_type == 'elevation':
        error_azi = np.abs([(gt_real[i, 1] - pred_angle[i][1]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    elif angle_type == 'tilt':
        error_azi = np.abs([(gt_real[i, 2] - pred_angle[i][2]) % 360 for i in range(batch_size)])
        error_azi[error_azi > 180] = 360 - error_azi[error_azi > 180]
    return error_azi


def mean(x, key):
    if key == 0:
        return np.clip(x, -45, 45) * np.abs(np.clip(x, -45, 45)) / 45.0
    elif key in [1, 2]:
        return np.clip(x, -15, 15) * np.abs(np.clip(x, -15, 15)) / 15.0


def sigma(x, k=10):
    return 15 + k * (1.5 - np.cos(2 * x / 180. * math.pi) - (1 - abs(x / 180.)) * 2)


def get_knn_visibility(pt_list):
    return np.array(pt_list)[:, 0, :, :2], np.array(pt_list)[:, 0, :, :2]


def tester():
    x = np.random.uniform(0, 360, size=(1, 3))


if __name__ == '__main__':
    tester()