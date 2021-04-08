import cv2 as cv
import numpy as np


for i in range(360):
    image = cv.imread("./test100/test{}.png".format(i), 0)
    img = cv.resize(image, (460, 460), interpolation=cv.INTER_AREA) #
    img = np.pad(img, 26, "constant", constant_values=0)
    img = np.repeat(img[:, :, np.newaxis], repeats=3, axis=-1)
    print(img.shape)
    cv.imwrite("./test90/test{}.png".format(i), img)

