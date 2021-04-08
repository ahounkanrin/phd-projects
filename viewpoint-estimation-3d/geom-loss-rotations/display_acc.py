import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt


testID = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.036Y.F.CT.58319", "SMIR.Body.037Y.M.CT.57613", "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.041Y.F.CT.57699", "SMIR.Body.057Y.F.CT.59693"]
imgs = [cv.imread("accuracy_{}.png".format(id))[..., ::-1] for id in testID]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 12)) #
fig.suptitle("Accuracy of viewpoint estimation for different CT-scans", fontsize=20)

ax[0][0].imshow(imgs[0])
ax[0][0].axis("off")
ax[0][1].imshow(imgs[1])
ax[0][1].axis("off")
ax[0][2].imshow(imgs[2])
ax[0][2].axis("off")

ax[1][0].imshow(imgs[3])
ax[1][0].axis("off")
ax[1][1].imshow(imgs[4])
ax[1][1].axis("off")
ax[1][2].imshow(imgs[5])
ax[1][2].axis("off")
plt.savefig("accuracy_all_test_imgs.png")
# for view_id in range(0, 180, 5):
#     imgs = [cv.imread("view{}/s{}.png".format(view_id, scale))[..., ::-1] for scale in scales]
#     fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5)) #

#     ax[0].imshow(imgs[0])
#     #ax[0][0].set_title(codes[0])
#     ax[0].axis('off')

#     ax[1].imshow(imgs[1])
#     #ax[0][1].set_title(codes[1])
#     ax[1].axis('off')

#     ax[2].imshow(imgs[2])
#     #ax[0][2].set_title(codes[2])
#     ax[2].axis('off')

#     ax[3].imshow(imgs[3])
#     #ax[0][2].set_title(codes[2])
#     ax[3].axis('off')

#     ax[4].imshow(imgs[4])
#     #ax[1][0].set_title(codes[5])
#     ax[4].axis('off')


#     #ax[1][3].imshow(imgs[7])
#     #ax[0][2].set_title(codes[2])
#     #ax[1][3].axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.savefig("./view{}/multiScaleHeatmap{}.png".format(view_id, view_id))
#     #plt.show()