import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_soft_classification_200x200 = 7.0
acc_soft_classification_200x200 = [0.0278, 0.4056, 0.6111, 0.6806, 0.6944, 0.7083, 0.7278, 0.7528, 0.7639, 0.7778, 0.8028, 0.8111]

medErr_soft_classification_3d = 34.5
acc_soft_classification_3d = [0.0139, 0.1528, 0.2528, 0.3583, 0.4278, 0.4667, 0.4806, 0.5083, 0.5250, 0.5444,  0.5694, 0.6000]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")

plt.plot(thresholds, acc_soft_classification_200x200, label="2D Input")
plt.plot(thresholds, acc_soft_classification_3d, label="3D Input")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("azimuth_accuracy.png")


medErrs = [medErr_soft_classification_200x200, medErr_soft_classification_3d]

plt.figure(figsize=[4, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="2D Input")
plt.bar(1, medErrs[1],  label="3D Input")

plt.xticks([0, 1], [])
plt.legend(loc="upper left")
plt.savefig("azimuth_mederr.png")

print("Done!")