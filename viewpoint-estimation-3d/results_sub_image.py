import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_full = 8.0  
acc_full = [0.0278, 0.4778, 0.6361, 0.7306, 0.7583, 0.7583, 0.7611, 0.7694, 0.7750, 0.7750, 0.7917, 0.8028]

medErr_central = 20.0000
acc_central = [0.0250,  0.2167, 0.3833, 0.4667,  0.5028, 0.5417,  0.5500, 0.5611, 0.5611, 0.5611, 0.5806, 0.5917]

medErr_up_10 = 12.0000
acc_up_10 = [0.0222, 0.2611, 0.4528, 0.5278, 0.5639, 0.5833, 0.5944, 0.6028, 0.6056, 0.6056, 0.6222, 0.6361]

medErr_up_20 = 10.0000
acc_up_20 = [0.0278, 0.2972, 0.5111, 0.6000, 0.6250, 0.6361, 0.6444, 0.6472, 0.6500, 0.6500, 0.6694, 0.6833]

medErr_up_50 = 8.0000 
acc_up_50 = [0.0222, 0.3389, 0.5917, 0.6694, 0.7000, 0.7139, 0.7194, 0.7306, 0.7333, 0.7333, 0.7417, 0.7472]

medErr_down_10 = 20.5000 
acc_down_10= [0.0306, 0.2028, 0.3333, 0.4611, 0.4917,  0.5250, 0.5361, 0.5417, 0.5444, 0.5500, 0.5722, 0.5833]

medErr_left_10 = 16.0000
acc_left_10= [0.0222, 0.2500, 0.4194, 0.4917, 0.5167, 0.5361, 0.5361, 0.5472, 0.5472, 0.5472, 0.5639, 0.5667]

medErr_right_10 = 21.0000
acc_right_10= [0.0139, 0.1611, 0.3278, 0.4333, 0.4944, 0.5278, 0.5444, 0.5556, 0.5611, 0.5722, 0.5917, 0.6000]


plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_full, label="full image")
plt.plot(thresholds, acc_central, label="sub-image: central")
plt.plot(thresholds, acc_up_10, label="sub-image: up-10-pixels")
plt.plot(thresholds, acc_up_20, label="sub-image: up-20-pixels")
plt.plot(thresholds, acc_up_50, label="sub-image: up-50-pixels")
plt.plot(thresholds, acc_down_10, label="sub-image: down-10-pixels")
plt.plot(thresholds, acc_left_10, label="sub-image: left-10-pixels")
plt.plot(thresholds, acc_right_10, label="sub-image: right-10-pixels")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_sub_images.png")



