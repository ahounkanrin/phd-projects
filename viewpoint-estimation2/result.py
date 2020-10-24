import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_soft_classification = 9.0
acc_soft_classification = [0.0222, 0.3194, 0.5861, 0.7083, 0.7500, 0.7611, 0.7639, 0.7722, 0.7750, 0.7917, 0.8000, 0.8083]

medErr_soft_classification_3d = 11.0
acc_soft_classification_3d = [0.0139, 0.2889, 0.4694, 0.5889, 0.6222, 0.6333, 0.6417, 0.6417, 0.6444, 0.6611, 0.6778, 0.6861]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")

plt.plot(thresholds, acc_soft_classification, label="2D Input")
plt.plot(thresholds, acc_soft_classification_3d, label="3D Input")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("azimuth_accuracy.png")


medErrs = [medErr_soft_classification, medErr_soft_classification_3d]

plt.figure(figsize=[4, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="2D Input")
plt.bar(1, medErrs[1],  label="3D Input")

plt.xticks([0, 1], [])
plt.legend(loc="upper left")
plt.savefig("azimuth_mederr.png")

print("Done!")