import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]
mederr = 2.0
acc = [0.2416, 0.6902, 0.7860, 0.8234, 0.8376,  0.8557, 0.8668, 0.8811, 0.8936, 0.9077, 0.9166, 0.9289, 0.9395, 0.9494, 0.9612, 0.9730, 0.9839, 0.9924, 1.0000] 

mederr1 = 8.0 
acc1 = [0.0404, 0.3528, 0.6035, 0.6656, 0.6880,  0.7150, 0.7350, 0.7639, 0.7930, 0.8158, 0.8298, 0.8510, 0.8690, 0.8936, 0.9158, 0.9327, 0.9530, 0.9795, 1.0000]

mederr2 = 3.0
acc2 = [0.1264, 0.7166, 0.7758, 0.8327, 0.8448, 0.8795, 0.8863, 0.8998, 0.9055, 0.9118, 0.9202,  0.9274, 0.9339, 0.9442, 0.9634, 0.9808, 0.9882, 0.9947, 1.0000]

mederr3 = 7.0
acc3 = [0.0456, 0.4002, 0.6448, 0.7398, 0.7783, 0.8004, 0.8248, 0.8441, 0.8682, 0.8887, 0.9035, 0.9228,  0.9400, 0.9484, 0.9568, 0.9695, 0.9867, 0.9907, 1.0000] 

mederr4 = 6.0
acc4 = [0.0654, 0.4808, 0.5731, 0.6081, 0.6335, 0.6630, 0.6853, 0.7179, 0.7425, 0.7798, 0.7971, 0.8233, 0.8531, 0.8717, 0.9011, 0.9307, 0.9568, 0.9794, 1.0000] 

mederr5 = 0
acc5 = [0.5055, 0.9991, 0.9995, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

mederr6 = 0
acc6 = [0.5230, 0.9994, 0.9996, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

mederr7 = 0
acc7 = [0.5530, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

mederr8 = 4.0000
acc8 = [0.0732, 0.5730, 0.6915, 0.7412, 0.7567, 0.7881, 0.8031, 0.8227, 0.8395, 0.8658, 0.8823, 0.9071, 0.9198, 0.9373, 0.9521, 0.9701, 0.9862, 0.9953, 1.0000]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1, label=testScans[0])
plt.plot(thresholds, acc2, label=testScans[1])
plt.plot(thresholds, acc3, label=testScans[2])
plt.plot(thresholds, acc4, label=testScans[3])
plt.plot(thresholds, acc5, label=testScans[4])
plt.plot(thresholds, acc6, label=testScans[5])
plt.plot(thresholds, acc7, label=testScans[6])
plt.plot(thresholds, acc8, label=testScans[7])

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc)
#plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_average.png")

print("Done!")