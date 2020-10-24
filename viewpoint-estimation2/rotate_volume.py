import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import get_view2, rotation_matrix, soft_label_encoding, one_hot_encoding
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage
from matplotlib import pyplot as plt

tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(True)

with tf.device('/gpu:0'):

    INPUT_SIZE = (200, 200)
    imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
    imgpath2 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"

    print("INFO: loading CT volumes...")
    tic1 = time.time()
    train_img3d = nib.load(imgpath1).get_fdata().astype(int)
    train_img3d = np.squeeze(train_img3d)
    train_img3d = train_img3d[:,:, :512]
    print("INFO: volumes loaded after {:.2f} seconds".format(time.time()-tic1))

    for theta in range(0, 100, 10):
        tic2 = time.time()
        #img = get_view2(train_img3d, theta, 0 , 0)
        img3d = tf.keras.preprocessing.image.apply_affine_transform(train_img3d, theta=theta, tx=0, ty=0, shear=0, zx=1, zy=1, 
                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=np.min(train_img3d), order=1)
        img = np.sum(img3d, axis=1)
        #img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
        #img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
        #plt.imshow(img, cmap="gray")
        #plt.savefig("img{}.png".format(str(theta)))
        print("rotation time {:.2f} seconds".format(time.time()-tic2))