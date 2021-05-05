import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

train_dir = "/scratch/hnkmah001/Datasets/ctfullbody/train-data/"
test_dir = "/scratch/hnkmah001/Datasets/ctfullbody/test-data/"
trainID = ["SMIR.Body.021Y.M.CT.57761", "SMIR.Body.058Y.M.CT.59468", "SMIR.Body.039Y.F.CT.58316"]
testID = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.036Y.F.CT.58319", "SMIR.Body.037Y.M.CT.57613", "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.041Y.F.CT.57699", "SMIR.Body.057Y.F.CT.59693"]
test_imgs = [cv.imread(test_dir+"test-{}/s100/test0.png".format(id))[..., ::-1] for id in testID]
train_imgs = [cv.imread(train_dir+"train-{}/s100/train0.png".format(id))[..., ::-1] for id in trainID]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 12)) #
fig.suptitle("Sample of test images", fontsize=20)

ax[0][0].imshow(test_imgs[0])
ax[0][0].axis("off")
ax[0][0].set_title(testID[0])
ax[0][1].imshow(test_imgs[1])
ax[0][1].axis("off")
ax[0][1].set_title(testID[1])
ax[0][2].imshow(test_imgs[2])
ax[0][2].axis("off")
ax[0][2].set_title(testID[2])

ax[1][0].imshow(test_imgs[3])
ax[1][0].axis("off")
ax[1][0].set_title(testID[3])
ax[1][1].imshow(test_imgs[4])
ax[1][1].axis("off")
ax[1][1].set_title(testID[4])
ax[1][2].imshow(test_imgs[5])
ax[1][2].axis("off")
ax[1][2].set_title(testID[5])
plt.savefig("sample_test_imgs.png")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 12)) #
fig.suptitle("Sample of train images", fontsize=20)

ax[0].imshow(train_imgs[0])
ax[0].axis("off")
ax[0].set_title(trainID[0])
ax[1].imshow(train_imgs[1])
ax[1].axis("off")
ax[1].set_title(trainID[1])
ax[2].imshow(train_imgs[2])
ax[2].axis("off")
ax[2].set_title(trainID[2])

plt.savefig("sample_train_imgs.png")



