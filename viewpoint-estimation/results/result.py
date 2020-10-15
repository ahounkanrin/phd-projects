import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]

medErr_regression = 45.3841
acc_regression = [0.0000, 0.0528, 0.1250, 0.1694, 0.2056, 0.2444, 0.3083, 0.3722, 0.4361, 0.4944, 0.5556, 0.6194]
medErr_knn =  23.5
acc_knn = [0.0056, 0.0778, 0.2111, 0.3222, 0.4500, 0.5167, 0.5778, 0.5972, 0.6083, 0.6139, 0.6333, 0.6417]
medErr_soft_classification = 9.0
acc_soft_classification = [0.0222, 0.3194, 0.5861, 0.7083, 0.7500, 0.7611, 0.7639, 0.7722, 0.7750, 0.7917, 0.8000, 0.8083]
medErr_hard_classification = 19.0
acc_hard_classification = [0.0139, 0.2083, 0.3667, 0.4556, 0.5056, 0.5389, 0.5611, 0.5917, 0.6083, 0.6167, 0.6250, 0.6333]
medErr_geom_loss = 8.0
acc_geom_loss = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]
medErr_soft_classification_300x300 = 7.0
acc_soft_classification_300x300 = [0.0222, 0.4222, 0.5833, 0.6639, 0.6806, 0.6806, 0.6889,  0.6972, 0.7139, 0.7167, 0.7167, 0.7417]
medErr_soft_classification_200x200 = 7.0
acc_soft_classification_200x200 = [0.0278, 0.4056, 0.6111, 0.6806, 0.6944, 0.7083, 0.7278, 0.7528, 0.7639, 0.7778, 0.8028, 0.8111]
medErr_soft_classification_100x100 = 15.5
acc_soft_classification_100x100 = [0.0167, 0.2528, 0.4083, 0.4944, 0.5083, 0.5167, 0.5250, 0.5333, 0.5389, 0.5500, 0.5556, 0.5833]

plt.figure(figsize=[8, 5])
plt.title("Azimuth estimation - Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")

plt.plot(thresholds, acc_regression, label="Regression")
plt.plot(thresholds, acc_knn, label="Nearest Neighbor")
plt.plot(thresholds, acc_soft_classification, label="Soft classification")
plt.plot(thresholds, acc_hard_classification, label="Hard classification")
plt.plot(thresholds, acc_geom_loss, label="Geometry-aware classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("azimuth_accuracy.png")

plt.figure(figsize=[8, 5])
plt.title("Azimuth estimation accuracy- Multiscale input size")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")

plt.plot(thresholds, acc_soft_classification, label=r"$400 \times 400$")
plt.plot(thresholds, acc_soft_classification_300x300, label=r"$300 \times 300$")
plt.plot(thresholds, acc_soft_classification_200x200, label=r"$200 \times 200$")
plt.plot(thresholds, acc_soft_classification_100x100, color="brown", label=r"$100 \times 100$")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("azimuth_accuracy_multiscale.png")


medErrs = [medErr_regression, medErr_knn, medErr_hard_classification, medErr_soft_classification, medErr_geom_loss]

medErrs_multiscale = [medErr_soft_classification, medErr_soft_classification_300x300, medErr_soft_classification_200x200, medErr_soft_classification_100x100]

plt.figure(figsize=[7, 4])
plt.title("Azimuth estimation - Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0], color="royalblue", label="Regression")
plt.bar(1, medErrs[1], color="orange", label="Nearest neighbor")
plt.bar(2, medErrs[2], color="r", label="Hard classification")
plt.bar(3, medErrs[3], color="g", label="Soft classification")
plt.bar(4, medErrs[4], color="darkviolet", label="Geometry-aware classification")

plt.xticks([0, 1, 2, 3, 4], [])
plt.legend(loc="upper right")
plt.savefig("azimuth_mederr.png")

plt.figure(figsize=[4, 4])
plt.title("Azimuth estimation Median Error - Multiscale input size")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs_multiscale[0], color="b", label=r"$400 \times 400$")
plt.bar(1, medErrs_multiscale[1], color="orange", label=r"$300\times 300$")
plt.bar(2, medErrs_multiscale[2], color="g", label=r"$200\times 200$")
plt.bar(3, medErrs_multiscale[3], color="brown", label=r"$100 \times 100$")
#plt.bar(3, medErrs_multiscale[3], color="g", label="Soft classification")
plt.xticks([0, 1, 2, 3], [])
plt.legend(loc="upper left")
plt.savefig("azimuth_mederr_multiscale.png")

print("Done!")