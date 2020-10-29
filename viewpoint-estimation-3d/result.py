import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_soft_classification = 7.0
acc_soft_classification = [0.0278, 0.4056, 0.6111, 0.6806, 0.6944, 0.7083, 0.7278, 0.7528, 0.7639, 0.7778, 0.8028, 0.8111]
medErr_soft_classification_3d = 6.0
acc_soft_classification_3d = [0.0194, 0.425, 0.6556, 0.7361, 0.7778, 0.8222, 0.8306, 0.8333, 0.8333, 0.8361, 0.8472, 0.8611]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")

plt.plot(thresholds, acc_soft_classification, label="2D Input")
plt.plot(thresholds, acc_soft_classification_3d, label="3D Input")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")


medErrs = [medErr_soft_classification, medErr_soft_classification_3d]

plt.figure(figsize=[4, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="2D Input")
plt.bar(1, medErrs[1],  label="3D Input")

plt.xticks([0, 1], [])
plt.legend(loc="upper right")
plt.savefig("mederr.png")

print("Done!")