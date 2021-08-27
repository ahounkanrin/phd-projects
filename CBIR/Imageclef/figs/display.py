import matplotlib.pyplot as plt
import cv2 as cv

img1 = cv.imread("traindata_hist.png", 0)
img2 = cv.imread("testdata_hist.png", 0)

n = 1  # how many digits we will display
#plt.figure(figsize=(30, 10))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img1)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(img2)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("data_hist.png")