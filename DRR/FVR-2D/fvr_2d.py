import nibabel as nib
import numpy as np
import time
import cv2 as cv
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2, ifft
from matplotlib import pyplot as plt
from scipy import ndimage

np.random.seed(0)
N = 400
def Rx(theta):
    x = theta * np.pi / 180.
    r11, r12 = np.cos(x), -np.sin(x)
    r21, r22 = np.sin(x), np.cos(x)
    return np.array([[r11, r12], [r21, r22]])

def rotate_line(line, rotationMatrix):
	return np.matmul(rotationMatrix, line)

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

if __name__ == "__main__":

	img = cv.imread("0_fvr.png", 0)
	#img = ndimage.rotate(img, 45, reshape=False)
	img = np.pad(img, N//2, "constant", constant_values=0)
	print(img.shape)
	imgShifted = fftshift(img)
	imgFFT = fftn(imgShifted)
	imgFFTShifted = fftshift(imgFFT)
	imgFFTShiftedReal = imgFFTShifted.real
	imgFFTShiftedImag = imgFFTShifted.imag
	print("2D FFT computed.")


	# Rotation and Interpolation of the projection slice from the 3D FFT volume

	x = np.linspace(-N+0.5, N-0.5, 2*N)
	y = np.linspace(-N+0.5, N-0.5, 2*N)
	#x = np.linspace(-N//2+0.5, N//2-0.5, N)
	#y = np.linspace(-N//2+0.5, N//2-0.5, N)
	projectionLine = np.array([[xi, 0.] for xi in x])
	projectionLine = np.reshape(projectionLine, (2*N, 2, 1))
	
	def render_view(theta, interpMethod):
		tic_rendering = time.time()
		rotationMatrix = Rx(theta)
		projectionSlice = np.squeeze(rotate_line(projectionLine, rotationMatrix))
		projectionSliceFFTReal = interpn(points=(x, y), values=imgFFTShiftedReal, xi=projectionSlice, method=interpMethod,
				                     bounds_error=False)   #, fill_value=0
		projectionSliceFFTImag = interpn(points=(x, y), values=imgFFTShiftedImag, xi=projectionSlice, method=interpMethod,
				                     bounds_error=False) 
		projectionSliceFFT = projectionSliceFFTReal + 1j * projectionSliceFFTImag			                     
		projection = fftshift(ifft(projectionSliceFFT))  
		projection = np.abs(projection)
		#projection = normalize(projection)
		toc_rendering = time.time()
		print("theta = {}\t {:.2f} seconds".format(theta, toc_rendering-tic_rendering))
		return projection
		
	img_x = np.sum(img, axis=1)
	img_x_mean = np.mean(img_x)
	img_y = np.sum(img, axis=0)
	img_y_mean = np.mean(img_y)
	projection_x_spline = render_view(0, "splinef2d") 
	projection_x_spline_mean = np.mean(projection_x_spline)
	projection_y_spline = render_view(90, "splinef2d")
	projection_y_spline_mean = np.mean(projection_y_spline)
	projection_x_linear = render_view(0, "linear") 
	projection_y_linear = render_view(90, "linear") 
	projection_x_nearest = render_view(0, "nearest") 
	projection_y_nearest = render_view(90, "nearest") 
	#img_x = normalize(img_x)
	#img_y = normalize(img_y)
	#projection_x = normalize(projection_x)
	#projection_y = normalize(projection_y)


	plt.figure(figsize=(8, 6))
	plt.title(r"Projection onto the x-axis (2X padding)")
	plt.plot(img_x, label="Direct projection")
	plt.plot(projection_x_nearest, label=r"FVR -- nearest neighbor")
	plt.plot(projection_x_linear, label=r"FVR -- linear")
	plt.plot(projection_x_spline, label=r"FVR -- spline")
	plt.legend(loc="upper left")
	plt.grid(True)
	plt.savefig("xprojection_padding_2X_spline.png")


	plt.figure(figsize=(8, 6))
	plt.title(r"Projection onto the y-axis (2X padding)")
	plt.plot(img_y, label="Direct projection")
	plt.plot(projection_y_nearest, label=r"FVR -- nearest neighbor")
	plt.plot(projection_y_linear, label=r"FVR -- linear")
	plt.plot(projection_y_spline, label=r"FVR -- spline")
	plt.legend(loc="upper left")
	plt.grid(True)
	plt.savefig("yprojection_padding_2X_spline.png")
	

	plt.figure(figsize=(20, 8))
	plt.subplot(2, 2, 1)
	plt.grid(True)
	plt.plot(img_x, label="Direct projection")
	plt.plot(projection_x_spline, label="FVR -- spline")
	#plt.grid(True)

	plt.subplot(2, 2, 2)
	plt.grid(True)
	plt.plot(img_x, label="Direct projection")
	plt.plot(img_x_mean - projection_x_spline_mean + projection_x_spline, label="FVR -- spline")
	
	plt.subplot(2, 2, 3)
	plt.grid(True)
	plt.plot(img_y, label="Direct projection")
	plt.plot(projection_y_spline, label="FVR -- spline")
	
	plt.subplot(2, 2, 4)
	plt.grid(True)
	plt.plot(img_y, label="Direct projection")
	plt.plot(img_y_mean - projection_y_spline_mean + projection_y_spline, label="FVR -- spline")
	plt.savefig("xyprojection_shifted_2X.png")
	
