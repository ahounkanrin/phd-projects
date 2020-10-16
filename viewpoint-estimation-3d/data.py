import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import get_view, rotation_matrix, soft_label_encoding, one_hot_encoding
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage
from matplotlib import pyplot as plt



INPUT_SIZE = (200, 200)
imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
imgpath2 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"

print("INFO: loading CT volumes...")
tic1 = time.time()
train_img3d = nib.load(imgpath1).get_fdata().astype(int)
train_img3d = np.squeeze(train_img3d)
train_img3d = train_img3d[:,:, :512]
print("INFO: volumes loaded after {:.2f} seconds".format(time.time()-tic1))

for theta in range(0, 360, 10):
    tic2 = time.time()
    img = get_view(train_img3d, theta)
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    plt.imshow(img, cmap="gray")
    plt.savefig("img{}.png".format(str(theta)))
    print("time {:.2f} seconds".format(time.time()-tic2))