import pandas as pd
import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2
from multiprocessing import Pool
from utils import quaternion_to_euler
from tqdm import tqdm
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

if __name__ == "__main__":
    

    df = pd.read_csv("test.csv", sep=",")
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
    
    df.to_csv("test.csv", sep=",", index=False)
