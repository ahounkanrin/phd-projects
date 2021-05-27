import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

#SMIR.Body.025Y.M.CT.57697

acc1x1x1_1 = [0.0083, 0.0667, 0.1389, 0.2528, 0.3472, 0.4361, 0.5139, 0.5806, 0.6472, 0.7472,  0.8111, 0.8333, 0.8667, 0.9083, 0.9667, 0.9806, 0.9833, 0.9889, 1.0000]
acc21x21x1_1 = [0.0250, 0.1556, 0.2778, 0.4139, 0.5361, 0.6333, 0.7000, 0.7472, 0.7833, 0.8167, 0.8583, 0.8972, 0.9194, 0.9528, 0.9889, 0.9972, 1.0000, 1.0000, 1.0000]
acc1x1x5_1 = [0.0194, 0.1667, 0.3361, 0.4861, 0.6000, 0.6806, 0.7250, 0.7639, 0.8167, 0.8472, 0.8750, 0.9139, 0.9278, 0.9556, 0.9944, 1.0000, 1.0000, 1.0000, 1.0000] 
acc21x21x5_1 = [0.0750, 0.4528, 0.6028, 0.7417, 0.8111, 0.8444, 0.8667, 0.8722, 0.8889, 0.9194, 0.9472, 0.9722, 0.9722, 0.9889, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.025Y.M.CT.57697")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_1, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_1, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_1, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_1, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.025Y.M.CT.57697.png")
# --------------------------------------------------------------------------------------------

#SMIR.Body.036Y.F.CT.58319 

acc1x1x1_2 = [0.0056, 0.0694, 0.1556, 0.2389, 0.3417, 0.4444, 0.5417, 0.6500, 0.7361, 0.7694, 0.8083, 0.8556, 0.8972, 0.9333, 0.9722, 0.9806, 0.9917, 0.9972, 1.0000]
acc21x21x1_2 = [0.0306, 0.2306, 0.3306, 0.4361, 0.5778, 0.6722, 0.7278, 0.8028, 0.8583, 0.8778, 0.9000, 0.9417, 0.9611, 0.9917, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
acc1x1x5_2 = [0.0389, 0.2611, 0.4250, 0.5556, 0.6667, 0.7528, 0.8083, 0.8333, 0.8611, 0.8889, 0.9111, 0.9333, 0.9583, 0.9806, 0.9972, 1.0000, 1.0000, 1.0000, 1.0000] 
acc21x21x5_2 = [0.1056, 0.5361, 0.6944, 0.7917, 0.8611, 0.9194, 0.9194, 0.9250, 0.9389, 0.9500, 0.9583, 0.9667, 0.9833, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.036Y.F.CT.58319")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_2, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_2, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_2, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_2, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.036Y.F.CT.58319.png")
#------------------------------------------------------------------------------------------------------------------------------

# SMIR.Body.037Y.M.CT.57613

acc1x1x1_3 = [0.0056, 0.0972, 0.1833, 0.2583, 0.3500, 0.4250, 0.5028, 0.5639, 0.6306, 0.6750, 0.7111, 0.7667, 0.7889, 0.7889, 0.8333, 0.8778, 0.9389, 0.9778, 1.0000]
acc21x21x1_3 = [0.0222, 0.2167, 0.3500, 0.4694, 0.5917, 0.6722, 0.7194, 0.7583, 0.8194, 0.8278, 0.8472, 0.8806, 0.9083, 0.9306, 0.9500, 0.9583, 0.9917, 1.0000, 1.0000]
acc1x1x5_3 = [0.0278, 0.2722, 0.3639, 0.4694, 0.6139, 0.7111, 0.7889, 0.8167, 0.8306, 0.8611, 0.8833, 0.9333, 0.9639, 0.9667, 0.9778, 0.9806, 0.9944, 1.0000, 1.0000] 
acc21x21x5_3 = [0.0778, 0.4667, 0.5889, 0.7000, 0.7806, 0.8667, 0.8917, 0.9222, 0.9472, 0.9583, 0.9583, 0.9833, 0.9917, 0.9972, 0.9972, 1.0000, 1.0000, 1.0000, 1.0000] 
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.037Y.M.CT.57613")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_3,  label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_3, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_3, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_3, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.037Y.M.CT.57613.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# SMIR.Body.040Y.M.CT.57768

acc1x1x1_4 = [0.0111, 0.1222, 0.2250, 0.3778, 0.4944, 0.5944, 0.6722, 0.7000, 0.7500, 0.7806, 0.8167, 0.8556, 0.8917, 0.9111, 0.9222, 0.9389, 0.9639, 0.9861, 1.0000]
acc21x21x1_4 = [0.0278, 0.2556, 0.4333, 0.5639, 0.6639, 0.7222, 0.7694, 0.7889, 0.8167, 0.8417, 0.8806, 0.8917, 0.9361, 0.9556, 0.9722, 0.9833, 1.0000, 1.0000, 1.0000]
acc1x1x5_4 = [0.0389, 0.3694, 0.5306, 0.6444, 0.7417, 0.7861, 0.8028, 0.8222, 0.8389, 0.8583, 0.8778, 0.8944, 0.9250, 0.9528, 0.9611, 0.9806, 1.0000, 1.0000, 1.0000] 
acc21x21x5_4 = [0.1111, 0.5528, 0.6861, 0.7889, 0.8250, 0.8444, 0.8444, 0.8611, 0.8667, 0.8833, 0.9056, 0.9250, 0.9639, 0.9722, 0.9750, 0.9972, 1.0000, 1.0000, 1.0000] 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.040Y.M.CT.57768")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_4,  label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_4, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_4, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_4, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.040Y.M.CT.57768.png")
# ----------------------------------------------------------------
# SMIR.Body.041Y.F.CT.57699

acc1x1x1_5 = [0.0167, 0.1528, 0.2417, 0.3278, 0.4306, 0.5444,  0.6306, 0.7000, 0.7278, 0.7972, 0.8333, 0.8778, 0.8889, 0.9028, 0.9250, 0.9389, 0.9639, 0.9833, 1.0000]
acc21x21x1_5 = [0.0417, 0.3306, 0.4917, 0.5750, 0.6694, 0.7250, 0.7667, 0.8111, 0.8556,  0.9056, 0.9194, 0.9306, 0.9556, 0.9611, 0.9667, 0.9778, 0.9917, 1.0000, 1.0000]
acc1x1x5_5 = [0.0306, 0.3028, 0.4639, 0.6028, 0.6917, 0.7556, 0.8111, 0.8444, 0.8861, 0.9139, 0.9306, 0.9472, 0.9556, 0.9611, 0.9639, 0.9722, 0.9889, 1.0000, 1.0000] 
acc21x21x5_5 = [0.1111, 0.5639, 0.7778, 0.8472, 0.8889, 0.9250, 0.9500, 0.9694, 0.9750, 0.9944, 0.9944, 0.9972, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.041Y.F.CT.57699")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_5,  label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_5, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_5, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_5, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.041Y.F.CT.57699.png")

# SMIR.Body.057Y.F.CT.59693

acc1x1x1_6 = [0.0056, 0.1000, 0.1917, 0.2806, 0.3833, 0.4917, 0.5722, 0.6056, 0.6528, 0.7028, 0.7528, 0.7889, 0.7944, 0.7972, 0.8472, 0.9000, 0.9167, 0.9528, 1.0000]
acc21x21x1_6 = [0.0306, 0.2917, 0.4556, 0.5500, 0.6389, 0.6889, 0.7444, 0.7639, 0.8056, 0.8194, 0.8528, 0.8750, 0.9000, 0.9167, 0.9389, 0.9556, 0.9778, 1.0000, 1.0000]
acc1x1x5_6 = [0.0361, 0.3444, 0.5222, 0.6417, 0.7111, 0.7556, 0.8167, 0.8583, 0.8722, 0.8861, 0.9250, 0.9417, 0.9639, 0.9861, 0.9972, 0.9972, 1.0000, 1.0000, 1.0000] 
acc21x21x5_6 = [0.1083, 0.5806, 0.7278, 0.7778, 0.8250, 0.8806, 0.9222, 0.9417, 0.9500, 0.9528, 0.9556, 0.9639, 0.9806, 0.9917, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 

plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.057Y.F.CT.59693")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_6,  label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_6, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_6, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_6, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.057Y.F.CT.59693.png")

# ---------------------------------------------------------------------------------------------------------

acc1x1x1 = np.mean(np.array([acc1x1x1_1, acc1x1x1_2, acc1x1x1_3, acc1x1x1_4, acc1x1x1_5, acc1x1x1_6]), axis=0)
acc21x21x1 = np.mean(np.array([acc21x21x1_1, acc21x21x1_2, acc21x21x1_3, acc21x21x1_4, acc21x21x1_5, acc21x21x1_6]), axis=0)
acc1x1x5 = np.mean(np.array([acc1x1x5_1, acc1x1x5_2, acc1x1x5_3, acc1x1x5_4, acc1x1x5_5, acc1x1x5_6]), axis=0)
acc21x21x5 = np.mean(np.array([acc21x21x5_1, acc21x21x5_2, acc21x21x5_3, acc21x21x5_4, acc21x21x5_5, acc21x21x5_6]), axis=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.figure(figsize=[8, 5])
plt.title("Accuracy of the CNN")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_combined.png")

print("Done!")
