import matplotlib.pyplot as plt
import numpy as np


thresholds = [theta for theta in range(0, 95, 10)]
az_el = [0.0000, 0.7802, 0.9116, 0.9498, 0.9656, 0.9776, 0.9848, 0.9902, 0.9959, 1.0000]

az_el_ip =  [0.0000, 0.6921, 0.8775, 0.9182, 0.9396, 0.9565, 0.9707, 0.9832, 0.9925, 1.0000] 

#quaternion = [0.0000, 0.7205, 0.8872, 0.9330, 0.9586, 0.9724, 0.9806, 0.9878, 0.9942, 1.0000]
# quaternion = [0.0000, 0.7720, 0.8620, 0.9020, 0.9260, 0.9500, 0.9690, 0.9800, 0.9910, 1.0000]
quaternion = [0.0000, 0.6561, 0.8214, 0.8787, 0.9151, 0.9395, 0.9572, 0.9713, 0.9856, 1.0000]
classification_500 = [0.0, 0.29725, 0.54625, 0.6125, 0.683, 0.73675, 0.7845, 0.855, 0.93075, 1.0]
#classification_1000 = [0.7321, 0.9646, 0.9719, 0.9735, 0.9754, 0.9794, 0.9841, 0.9894, 0.9955, 1.0000]
classification_1000 = [0.0000, 0.4773, 0.6944, 0.7416, 0.7800, 0.8153, 0.8542, 0.9106, 0.9617, 1.0000]
#classification_2000 = [0.7126, 0.9494, 0.9596, 0.9647, 0.9696, 0.9757, 0.9824, 0.9873, 0.9944, 1.0000]
classification_2000 = [0.0000, 0.6637, 0.7906, 0.8233, 0.8548, 0.8827, 0.9091, 0.9354, 0.9690, 1.0000] 
#classification_5000 = [0.6325, 0.9071, 0.9240, 0.9354, 0.9489, 0.9625, 0.9745, 0.9831, 0.9920, 1.0000]
classification_5000 = [0.0000, 0.8433, 0.8890, 0.9090, 0.9276, 0.9444, 0.9597, 0.9717, 0.9849, 1.0000]
classification_20000 = [0.0000, 0.8434, 0.8760, 0.8948, 0.9129, 0.9313, 0.9485, 0.9656, 0.9825, 1.0000]

reg_1k = [0.0000, 0.4445, 0.6910, 0.7931, 0.8521, 0.8955, 0.9291, 0.9625, 0.9829, 1.0000]
reg_2k = [0.0000, 0.4234, 0.6911, 0.8027, 0.8682, 0.9141, 0.9477, 0.9684, 0.9856, 1.0000]
reg_5k = [0.0000, 0.6561, 0.8214, 0.8787, 0.9151, 0.9395, 0.9572, 0.9713, 0.9856, 1.0000]
reg_20k = [0.0000, 0.7720, 0.8620, 0.9020, 0.9260, 0.9500, 0.9690, 0.9800, 0.9910, 1.0000]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])

#plt.plot(thresholds, az_el, label="Azimuth-Elevation-to-quaternions")
#plt.plot(thresholds, az_el_ip, label="Regression: Euler-to-quaternions")
# plt.plot(thresholds, quaternion, label="Regression")
# plt.plot(thresholds, classification_500, label="Classification-500-classes")
plt.plot(thresholds, classification_1000, label="1000 quaternions")
plt.plot(thresholds, classification_2000, label="2000 quaternions")
plt.plot(thresholds, classification_5000, label="5000 quaternions")
plt.plot(thresholds, classification_20000, label="20000 quaternions")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_quaternion_classification.png")

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])

plt.plot(thresholds, reg_1k, label="1000 quaternions")
plt.plot(thresholds, reg_2k, label="2000 quaternions")
plt.plot(thresholds, reg_5k, label="5000 quaternions")
plt.plot(thresholds, reg_20k, label="20000 quaternions")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_quaternion_regression.png")


acc_theta_reg = [0.79, 0.80, 0.88, 0.90]
acc_theta_class = [0.74, 0.82, 0.91, 0.89]

med_error_reg = [11.44, 11.95, 6.9, 5.05]
med_error_class = [10.51, 7.31, 4.84, 3.65]

labels = ["1K", "2K", "5K", "20K"]
x = np.arange(len(labels))  # the label locations
width = 0.35 # the width of the bars

fig, ax = plt.subplots(figsize=[8, 5])
rects1 = ax.bar(x - width/2, acc_theta_reg, width, label='Regression')
rects2 = ax.bar(x + width/2, acc_theta_class, width, label='Classification')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xlabel("Number of sampled quaternions.")
# ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend()

ax.bar_label(rects1)
ax.bar_label(rects2)

fig.tight_layout()

plt.savefig("barplot_accuracy.png")

fig, ax = plt.subplots(figsize=[8, 5])
rects1 = ax.bar(x - width/2, med_error_reg, width, label='Regression')
rects2 = ax.bar(x + width/2, med_error_class, width, label='Classification')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error (degrees)')
ax.set_xlabel("Number of sampled quaternions")
# ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
# ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend()

ax.bar_label(rects1)
ax.bar_label(rects2)

fig.tight_layout()

plt.savefig("barplot_error.png")
