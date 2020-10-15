import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]

mederr_sigma1 = 11.0
acc_sigma1 = [0.0000, 0.2972, 0.4889, 0.5278, 0.5583, 0.5778, 0.5861, 0.5944, 0.5944, 0.6000, 0.6056, 0.6333]

#mederr_sigma2 = 10.0
#acc_sigma2 = [0.0000, 0.3306, 0.5083, 0.6139, 0.6806, 0.7028, 0.7139, 0.7278, 0.7500, 0.7500, 0.7528, 0.7583]

#mederr_sigma3 = 10.0 
#acc_sigma3 = [0.0000, 0.2833, 0.5111, 0.6500, 0.6806, 0.7111, 0.7167, 0.7333, 0.7444, 0.7583, 0.7667, 0.7750]

mederr_sigma4 = 9.0
acc_sigma4 = [0.0000, 0.2889, 0.5611, 0.6528, 0.6806, 0.7000, 0.7139, 0.7389, 0.7444, 0.7583, 0.7750, 0.8028]

#mederr_sigma5 = 10.5
#acc_sigma5 = [0.0000, 0.3333, 0.4917, 0.5806, 0.6444, 0.6889, 0.6972, 0.6972, 0.7083, 0.7278, 0.7500, 0.7556]

#mederr_sigma6 = 10.0
#acc_sigma6 = [0.0000, 0.3528, 0.5167, 0.6139, 0.6528, 0.6806, 0.6917, 0.7222, 0.7472, 0.7806, 0.7861, 0.7972]

mederr_sigma7 = 8.0
acc_sigma7 = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]

#mederr_sigma8 = 12.0
#acc_sigma8 = [0.0000, 0.2778, 0.4583, 0.5611, 0.6250, 0.6361, 0.6694, 0.6944, 0.7139, 0.7250, 0.7306, 0.7472]

mederr_sigma10 = 9.5
acc_sigma10 = [0.0000, 0.3500, 0.5111, 0.5944, 0.6250, 0.6528, 0.6806, 0.7028, 0.7111, 0.7361, 0.7750, 0.8028]

mederr_sigma15 = 20.0
acc_sigma15 = [0.0000, 0.1722, 0.3083, 0.4278, 0.5000, 0.5694, 0.6028, 0.6278, 0.6361, 0.6361, 0.6444, 0.6611]


mederr_sigma0_5 = 16.0
acc_sigma0_5 = [0.0000, 0.2389, 0.4000, 0.4833, 0.5389, 0.5833, 0.6111, 0.6389, 0.6639, 0.6861, 0.6944, 0.7139]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Maximum error (degrees)")
plt.plot(thresholds, acc_sigma0_5, label=r"$\sigma = 0.5$")
plt.plot(thresholds, acc_sigma1, label=r"$\sigma = 1$")
#plt.plot(thresholds, acc_sigma2, label=r"$\sigma = 2$")
#plt.plot(thresholds, acc_sigma3, label=r"$\sigma = 3$")
plt.plot(thresholds, acc_sigma4, label=r"$\sigma = 4$")
#plt.plot(thresholds, acc_sigma5, label=r"$\sigma = 5$")
#plt.plot(thresholds, acc_sigma6, label=r"$\sigma = 6$")
plt.plot(thresholds, acc_sigma7, label=r"$\sigma = 7$")
#plt.plot(thresholds, acc_sigma8, label=r"$\sigma = 8$")
plt.plot(thresholds, acc_sigma10, label=r"$\sigma = 10$")
plt.plot(thresholds, acc_sigma15, label=r"$\sigma = 15$")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")


medErrs = [ mederr_sigma0_5, mederr_sigma1, mederr_sigma4, mederr_sigma7, mederr_sigma10, mederr_sigma15]

plt.figure(figsize=[5, 5])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label=r"$\sigma = 0.5$")
plt.bar(1, medErrs[1],  label=r"$\sigma = 1$")
plt.bar(2, medErrs[2],  label=r"$\sigma = 4$")
plt.bar(3, medErrs[3],  label=r"$\sigma = 7$")
plt.bar(4, medErrs[4],  label=r"$\sigma = 10$")
plt.bar(5, medErrs[5],  label=r"$\sigma = 15$")

plt.xticks([0, 1, 2, 3, 4, 5], [])
plt.legend(loc="lower left")
plt.savefig("mederr.png")

print("Done!")