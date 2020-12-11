import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2

np.random.seed(0)
min_ctnumber = -1024
def Rx(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    x = theta * np.pi / 180.
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
    # Load ct volume
    imgpath = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
    N = 512 
    print("INFO: loading CT volume...")
    tic_load = time.time()
    ctVolume = nib.load(imgpath).get_fdata().astype(int)
    ctVolume = np.squeeze(ctVolume)
    voi = ctVolume[:,:, :N] # Extracts volume of interest (512 x 512 x 512) from the full body ct volume (512 x 512 x 3000)
    voi = voi - min_ctnumber # shift to avoid negative CT numbers
    voi = np.pad(voi, N//2, "constant", constant_values=0)
    toc_load = time.time()
    print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

    tic_fft = time.time()
    voiShifted = fftshift(voi)
    del voi
    voiFFT = fftn(voiShifted)
    del voiShifted
    voiFFTShifted = fftshift(voiFFT)
    del voiFFT
    toc_fft = time.time()
    print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


    # Rotation and Interpolation of the projection slice from the 3D FFT volume

    x = np.linspace(-N+0.5, N-0.5, 2*N)
    y = np.linspace(-N+0.5, N-0.5, 2*N)
    z = np.linspace(-N+0.5, N-0.5, 2*N)


    projectionPlane = np.array([[xi, 0, zi] for xi in x for zi in z])
    projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

    def render_view(viewpoint):
        theta_x = 0 #np.random.randint(-10, 10)
        theta_y = 0 #np.random.randint(-10, 10)
        theta_z = viewpoint[0]
        tx = viewpoint[1]
        ty = viewpoint[2]
        tic_rendering = time.time()
        rotationMatrix = Rx(-theta_x) @ Ry(-theta_y) @ Rz(-theta_z) # Rotate the projection plane clockwise to get a counter clockwise projections of the ct volume
        projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
        projectionSliceFFT = interpn(points=(x, y, z), values=voiFFTShifted, xi=projectionSlice, method="linear",
                                     bounds_error=False)      
        img = np.abs(fftshift(ifft2(projectionSliceFFT)))
        img = img[N//2:N+N//2, N//2:N+N//2]
        img = normalize(img)
        img = img[54+tx:454+tx, 63+ty:463+ty]
        cv.imwrite("{}_fvr.png".format(theta_z), img)
        toc_rendering = time.time()
        print("theta = {}\t {:.2f} seconds".format(theta_z, toc_rendering-tic_rendering))
        

    viewpoints = [(i, 0, 0) for i in range(0, 100, 10)]
    for viewpoint in viewpoints:
        render_view(viewpoint)

