import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/data-ry/normal/train-val/SMIR.Body.021Y.M.CT.57761/s100/"
count = 1
fig = plt.figure(figsize=(14, 8))
#fig.suptitle("Rotation about the z-axis")
for theta in range(0, 180, 30):
        fig.add_subplot(2, 3, count)
        plt.axis("off")
        img = cv.imread(img_dir+str(theta)+".png")
        img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA) 
        plt.title(r"$\theta_y$ = {} deg.".format(theta))
        plt.imshow(img)
        plt.gray()
        count += 1
    # print("Test image index:", img_index)
    # print("Closest training image indexes:", least_n)

plt.savefig("rotation_thetay.png")
print("Done.")
