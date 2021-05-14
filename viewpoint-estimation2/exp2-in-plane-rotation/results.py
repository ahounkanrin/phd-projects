import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]




acc1 = [0.03333333333333333, 0.15833333333333333, 0.21666666666666667, 0.2833333333333333, 0.33611111111111114, 
        0.3638888888888889, 0.39166666666666666, 0.425, 0.45555555555555555, 0.49166666666666664, 0.5166666666666667, 
        0.5583333333333333, 0.5805555555555556, 0.6194444444444445, 0.6666666666666666, 0.8055555555555556, 0.9888888888888889, 
        1.0, 1.0]


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