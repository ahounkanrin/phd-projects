import nibabel as nib
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
import cv2 as cv

def rotation_matrix(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    r = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return r

def rotate_vector(vector, theta):
    rotatedVector = np.matmul(rotation_matrix(theta), vector)
    return rotatedVector

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val)/(max_val - min_val)
    img = 255 * img 
    img = 255 - img  # use 255 for background pixel values
    img = np.uint8(img)
    #img = img[54:454, 63:463]
    return img


# Load ct volume
imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
N = 512 
print("INFO: loading CT volume...")
tic_load = time.time()
ctVolume = nib.load(imgpath).get_fdata().astype(int)
ctVolume = np.squeeze(ctVolume)
voi = ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
toc_load = time.time()
print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

tic_fft = time.time()
voiShifted = np.fft.fftshift(voi)
voiFFT = np.fft.fftn(voiShifted)
voiFFTShifted = np.fft.fftshift(voiFFT)
toc_fft = time.time()
print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


# Rotation and Interpolation of the projection slice from the 3D FFT volume

x = np.linspace(-N/2+0.5, N/2-0.5, N)
y = np.linspace(-N/2+0.5, N/2-0.5, N)
z = np.linspace(-N/2+0.5, N/2-0.5, N)

my_interpolating_function = RegularGridInterpolator((x, y, z), voiFFTShifted)

for theta in range(0, 360, 10):
    tic_slice = time.time()
    ptsProjectionPlane = np.array([rotate_vector(np.array([x[i], 0, z[j]]), theta=theta) 
                                   for i in range(N) for j in range(N)])
       
    ptsProjectionPlaneInterpolated =  my_interpolating_function(ptsProjectionPlane) 
    projectionPlane = np.reshape(ptsProjectionPlaneInterpolated, (N,N), order="F")     
    projectionPlaneIFFT = np.abs(np.fft.ifft2(projectionPlane))
    img = np.fft.fftshift(projectionPlaneIFFT)
    img = normalize(img)

    #plt.imshow(img, cmap="gray")
    #plt.savefig("{}.png".format(theta)) 
    cv.imwrite("{}.png".format(theta), img)
    toc_slice = time.time()
    print("theta = {}\t time: {:.2f} seconds".format(theta, toc_slice-tic_slice))