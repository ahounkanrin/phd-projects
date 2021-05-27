import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 21, 1)]


medErr_rfn_0_10 = 4.0000
acc_rfn_0_10 = [0.0476, 0.1429, 0.1905, 0.3810, 0.4762, 0.5714, 0.7143, 0.7143, 0.7619, 0.8571, 0.9048, 0.9524, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

medErr_rfn_0_20 = 11.0000
acc_rfn_0_20 = [0.0000, 0.0000, 0.0000, 0.0488, 0.0732, 0.0976, 0.1463, 0.2195, 0.2683, 0.4146, 0.4878, 0.5366, 0.5854, 0.6585, 0.7073, 0.7073, 0.7561, 0.8049, 0.8537, 0.8780, 0.9268]

medErr_rfn_0_30 = 8.0000
acc_rfn_0_30 = [0.0328, 0.0656, 0.1148, 0.1803, 0.2131, 0.2623, 0.3607,  0.4426, 0.4590, 0.5410, 0.5410, 0.5410, 0.5410, 0.5574, 0.6230, 0.6557, 0.6885, 0.7213, 0.7377, 0.7541, 0.7705]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_rfn_0_10, label="viewpoint range: -10 to 10")
plt.plot(thresholds, acc_rfn_0_20, label="viewpoint range: -20 to 20")
plt.plot(thresholds, acc_rfn_0_30, label="viewpoint range: -30 to 30")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_refinement2.png")



