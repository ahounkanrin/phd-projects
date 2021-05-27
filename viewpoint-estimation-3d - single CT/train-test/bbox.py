import cv2 as cv
import numpy as np

img = cv.imread("train_0.png")
image = cv.rectangle(img, (56, 56), (456, 456), (0, 0, 255), 2)
cv.imshow("image", image)
cv.waitKey(0)
cv.imwrite("train_img.png", image)