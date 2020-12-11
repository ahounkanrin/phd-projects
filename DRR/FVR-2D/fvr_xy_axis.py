import cv2 as cv
from matplotlib import pyplot as plt 
import numpy as np



plt.figure(figsize=(25, 25))
       
plt.subplot(2, 2, 1)
img1 = cv.imread("xprojection_padding_1X_spline.png")
plt.imshow(img1)
#plt.title("FVR: theta = "+str(i*30))
plt.axis("off")

plt.subplot(2, 2, 2)
img2 = cv.imread("xprojection_padding_2X_spline.png")
plt.imshow(img2)
#plt.title("PP: theta = "+str(i*30))
plt.axis("off")

plt.subplot(2, 2, 3)
img3 = cv.imread("yprojection_padding_1X_spline.png")
plt.imshow(img3)
#plt.title("FVR: theta = "+str(i*30))
plt.axis("off")

plt.subplot(2, 2, 4)
img2 = cv.imread("yprojection_padding_2X_spline.png")
plt.imshow(img2)
#plt.title("PP: theta = "+str(i*30))
plt.axis("off")

plt.savefig("fvr_interpolationMethods.png")
