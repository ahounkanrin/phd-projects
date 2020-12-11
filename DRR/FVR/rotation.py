import cv2 as cv
import tensorflow as tf
from scipy import ndimage
from matplotlib import pyplot as plt 


img = cv.imread("ry0.png")
for theta in range(0, 181, 30):
    img_rotated = ndimage.rotate(img, -theta, reshape=False)
    cv.imwrite("ry{}_2.png".format(theta), img_rotated)

n = 7
plt.figure(figsize=(30, 8))
for i in range(n):
    
    plt.subplot(2, n, i+1)
    img_3d = cv.imread("ry{}.png".format(i*30))
    plt.imshow(img_3d)
    plt.gray()

    plt.subplot(2, n, i+1+n)
    img_2d = cv.imread("ry{}_2.png".format(i*30))
    plt.imshow(img_2d)
    plt.gray()
    plt.title("theta = "+str(i*30))

plt.savefig("rotation.png")
