import nibabel as nib
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
import cv2 as cv


def rotation_matrix(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
   
def rotate_plane(plane, rotationMatrix):
	return np.matmul(rotationMatrix, plane)

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val)/(max_val - min_val)
    img = 255 * img 
    img = 255 - img  # use 0 for background pixel values
    return np.uint8(img)

# Load ct volume
imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
N = 512 
print("INFO: loading CT volume...")
tic_load = time.time()
ctVolume = nib.load(imgpath).get_fdata().astype(int)
ctVolume = np.squeeze(ctVolume)
voi = ctVolume[:,:, :N] # Extracts volume of interest (512 x 512 x 512) from the full body ct volume (512 x 512 x 3000)
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
projectionPlane = np.array([[xi, 0, zi] for xi in x for zi in z])
projectionPlane = np.reshape(projectionPlane, (N, N, 3, 1), order="F")

def render_view(viewpoint):
    theta = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    tic_rendering = time.time()
    rotationMatrix = rotation_matrix(theta)
    projectionSlice = rotate_plane(projectionPlane, rotationMatrix)
    projectionSlice = np.squeeze(projectionSlice)
    projectionSliceInterpolated =  my_interpolating_function(projectionSlice)     
    projectionSliceIFFT = np.abs(np.fft.ifft2(projectionSliceInterpolated))
    img = np.fft.fftshift(projectionSliceIFFT)
    img = normalize(img)
    img = img[54+tx:454+tx, 63+ty:463+ty]
    cv.imwrite("{}.png".format(theta), img)
    toc_rendering = time.time()
    print("theta = {}\t {:.2f} seconds".format(theta, toc_rendering-tic_rendering))
    #with open("timelogs_fvr.txt", "a") as f:
	#    print("theta = {}\t {:.2f} seconds".format(theta, toc_rendering-tic_rendering), file=f)


viewpoints = [(i, 0, 0) for i in range(0, 360, 10)]
for viewpoint in viewpoints:
    render_view(viewpoint)
