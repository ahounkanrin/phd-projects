import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

test_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/normals/test/"
testID = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]
test_imgs = [cv.imread(test_dir+"{}/s100/0.png".format(id))[..., ::-1] for id in testID]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 12)) #
fig.suptitle("Front views from the test CT scans", fontsize=20)

ax[0][0].imshow(test_imgs[0])
ax[0][0].axis("off")
ax[0][0].set_title(testID[0])
ax[0][1].imshow(test_imgs[1])
ax[0][1].axis("off")
ax[0][1].set_title(testID[1])
ax[0][2].imshow(test_imgs[2]) 
ax[0][2].axis("off")
ax[0][2].set_title(testID[2])
ax[0][3].imshow(test_imgs[3])
ax[0][3].axis("off")
ax[0][3].set_title(testID[3])

ax[1][0].imshow(test_imgs[4])
ax[1][0].axis("off")
ax[1][0].set_title(testID[4])
ax[1][1].imshow(test_imgs[5])
ax[1][1].axis("off")
ax[1][1].set_title(testID[5])
ax[1][2].imshow(test_imgs[6])
ax[1][2].axis("off")
ax[1][2].set_title(testID[6])
ax[1][3].imshow(test_imgs[7])
ax[1][3].axis("off")
ax[1][3].set_title(testID[7])
plt.savefig("test_imgs.png")



