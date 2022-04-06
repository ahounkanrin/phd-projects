import pandas as pd
import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2
from multiprocessing import Pool
from tqdm import tqdm
import math
np.random.seed(0)

def Rx(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    x = theta #* np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def rotate_plane(plane, rotationMatrix):
	return np.matmul(rotationMatrix, plane)

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

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

if __name__ == "__main__":
    

    df = pd.read_csv("test500.csv", sep=",")
    qw = df["qw"].tolist()
    qx = df["qx"].tolist()
    qy = df["qy"].tolist()
    qz = df["qz"].tolist()
    xvalues, yvalues, zvalues = [], [], []
    for i in tqdm(range(len(qw))):
        pitch, yaw, roll = quaternion_to_euler([qw[i], qx[i], qy[i], qz[i]])
        x = np.sin(pitch + np.pi/2) * np.cos(roll + np.pi/2)
        y = np.sin(pitch + np.pi/2) * np.sin(roll + np.pi/2)
        z = np.cos(pitch + np.pi/2)
        xvalues.append(x)
        yvalues.append(y)
        zvalues.append(z)
    
    df["x"] = xvalues
    df["y"] = yvalues
    df["z"] = zvalues
    
    df.to_csv("test500.csv", sep=",", index=False)
