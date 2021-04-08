import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 11, 1)]


medErr_rfn_20 = 6.0
acc_rfn_20 = [0.1000, 0.2000, 0.2500, 0.4000, 0.4500, 0.4500, 0.5500, 0.6500, 0.7000, 0.9000, 0.9500]

medErr_rfn_40 = 6.0
acc_rfn_40 = [0.0500, 0.1000, 0.1500, 0.2500, 0.3000, 0.4000, 0.5500, 0.7500, 0.8000, 1.0000, 1.0000 ]

medErr_rfn_60 = 5.50
acc_rfn_60 = [0.0000, 0.1000, 0.1500, 0.3000, 0.4500, 0.4500, 0.5500, 0.6000, 0.6500, 0.6500, 0.8000]

medErr_rfn_80 = 4.0
acc_rfn_80 = [0.0500, 0.1000, 0.2000, 0.4000, 0.5500, 0.6000, 0.6000, 0.6500, 0.6500, 0.6500, 0.7000]

medErr_rfn_100 = 6.5000 
acc_rfn_100 = [0.0500, 0.1000, 0.2500, 0.3000, 0.4000, 0.4500, 0.5000, 0.5500, 0.5500, 0.6500, 0.7000]

medErr_rfn_120 = 6.0000
acc_rfn_120 = [0.0000, 0.0500, 0.2000, 0.3500, 0.4500, 0.4500, 0.5000, 0.7000, 0.7000, 0.8500, 0.8500]

medErr_rfn_140 = 2.0000
acc_rfn_140 = [0.0500, 0.3500, 0.4500, 0.6000, 0.6000, 0.6500, 0.8500, 0.8500, 0.8500, 0.9000, 0.9500]

medErr_rfn_160 = 6.0000
acc_rfn_160 = [0.0000, 0.0000, 0.1500, 0.2000, 0.2500, 0.4500, 0.5500, 0.7500, 0.8000, 0.9500, 1.0000]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_rfn_20, label="refinement: 0 - 20")
plt.plot(thresholds, acc_rfn_40, label="refinement: 20 - 40")
plt.plot(thresholds, acc_rfn_60, label="refinement: 40 - 60")
plt.plot(thresholds, acc_rfn_80, label="refinement: 60 - 80")
plt.plot(thresholds, acc_rfn_100, label="refinement: 80 - 100")
plt.plot(thresholds, acc_rfn_120, label="refinement: 100 - 120")
plt.plot(thresholds, acc_rfn_140, label="refinement: 120 - 140")
plt.plot(thresholds, acc_rfn_160, label="refinement: 140 - 160")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_refinement.png")



