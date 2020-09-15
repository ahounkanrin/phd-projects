#import nibabel as nib
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import cv2 as cv
import os

eng = matlab.engine.start_matlab()

DATA_DIR = "/home/anicet/Datasets/ctfullbody/"
#imgpath = "/home/anicet/Datasets/ctfullbody/SMIR.Body.033Y.M.CT.57764/SMIR.Body.033Y.M.CT.57764.nii"
#ctscan = nib.load(imgpath)
#data = ctscan.get_fdata()
#data = np.squeeze(data)
#data_up = data[:,:,:512]

for dirs in os.listdir(DATA_DIR):
    img = eng.projection2d(DATA_DIR+dirs+"/"+dirs+".nii", 45, "x")
    img = np.array(img).astype("uint8")
    img = cv.resize(img, (200, 200), interpolation=cv.INTER_AREA)
    plt.imshow(img, cmap='gray')
    plt.show()
