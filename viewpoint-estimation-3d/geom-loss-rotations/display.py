import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt


scales = ["80", "90", "100", "110", "120"]
for view_id in range(0, 180, 5):
    imgs = [cv.imread("view{}/s{}.png".format(view_id, scale))[..., ::-1] for scale in scales]
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5)) #

    ax[0].imshow(imgs[0])
    #ax[0][0].set_title(codes[0])
    ax[0].axis('off')

    ax[1].imshow(imgs[1])
    #ax[0][1].set_title(codes[1])
    ax[1].axis('off')

    ax[2].imshow(imgs[2])
    #ax[0][2].set_title(codes[2])
    ax[2].axis('off')

    ax[3].imshow(imgs[3])
    #ax[0][2].set_title(codes[2])
    ax[3].axis('off')

    ax[4].imshow(imgs[4])
    #ax[1][0].set_title(codes[5])
    ax[4].axis('off')


    #ax[1][3].imshow(imgs[7])
    #ax[0][2].set_title(codes[2])
    #ax[1][3].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("./view{}/multiScaleHeatmap{}.png".format(view_id, view_id))
    #plt.show()