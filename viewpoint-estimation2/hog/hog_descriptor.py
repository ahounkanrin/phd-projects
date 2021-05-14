import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import hog
from skimage import data, exposure
import tensorflow as tf

image = cv.imread("90_fvr.png", 0)
image = cv.resize(image, (200, 200), interpolation=cv.INTER_AREA)
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True, multichannel=None)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.savefig("90_hog8.png.")
#plt.show()

print(fd.ravel().shape)


# inputs = tf.keras.Input(shape=(20736,)) # (86436,) | 4356,)
# x = tf.keras.layers.Flatten()(inputs)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dropout(rate=0.2)(x)
# outputs = tf.keras.layers.Dense(360, activation='softmax')(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.build((None, 20736,))
# model.summary()
