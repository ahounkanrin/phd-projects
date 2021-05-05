import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]


acc1 = [0.2695, 0.7317, 0.8061, 0.8354, 0.8461, 0.8637, 0.8728, 0.8863, 0.8972, 0.9105, 0.9185, 0.9298, 0.9394, 0.9495, 0.9618, 
        0.9735, 0.9835, 0.9927, 1.0000]
acc2 = []
acc3 = [] 
acc4 = [] 


plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1, label=r"Reference")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")

print("Done!")