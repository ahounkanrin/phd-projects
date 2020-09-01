import numpy as np
from matplotlib import pyplot as plt

thresholds = [theta for theta in range(0, 60, 5)]

regression = [0.0000, 0.0639,  0.1278 , 0.1750, 0.2167, 0.2500, 0.3028 , 0.3694 , 0.4333 , 0.5028, 0.5722, 0.6111] 
knn = [0.0083, 0.0806, 0.2222, 0.3472, 0.4583, 0.5222, 0.5778, 0.6,  0.6083, 0.6222, 0.6333, 0.6416]
soft_classification = [0.0306, 0.3444, 0.5889, 0.7167, 0.7528, 0.7611, 0.7639, 0.7722, 0.7750, 0.7944, 0.8000,0.8111]
hard_classification = [0.0306, 0.2111, 0.3333, 0.4028, 0.4611, 0.5028, 0.5222, 0.55, 0.5528, 0.5639, 0.5778, 0.5889]
plt.figure(figsize=[10, 7])
plt.title("Azimuth angle estimation")
plt.ylabel("Accuracy")
plt.xlabel("Azimuth angle error (degrees)")

plt.plot(thresholds, regression, label="Regression")
plt.plot(thresholds, knn, label="KNN")
plt.plot(thresholds, soft_classification, label="Soft-classification")
plt.plot(thresholds, hard_classification, label="Hard-classification")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("azimuth_accuracy.png")