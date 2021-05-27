import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

#SMIR.Body.025Y.M.CT.57697 (ct3)

acc1x1x1_1 = [0.0333, 0.2778, 0.4528, 0.5583, 0.6222, 0.6583, 0.6861, 0.7361, 0.7944, 0.8111, 0.8250, 0.8444, 0.8611, 0.8861, 0.9111, 0.9472, 0.9694, 0.9944, 1.0000]
acc21x21x1_1 = [0.1056, 0.6306, 0.7750,  0.8500, 0.8944, 0.9333, 0.9694, 0.9806, 0.9944, 0.9944, 0.9944, 0.9944, 0.9944, 0.9944, 0.9944, 1.0000, 1.0000, 1.0000, 1.0000]
acc1x1x5_1 = [0.0861, 0.5917, 0.7778, 0.8444, 0.9000, 0.9194, 0.9222, 0.9500, 0.9639, 0.9639, 0.9639, 0.9639, 0.9667, 0.9722, 0.9722, 0.9861, 0.9972, 1.0000, 1.0000] 
acc21x21x5_1 = [0.2194, 0.8111, 0.9500, 0.9889, 0.9944, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
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

#SMIR.Body.036Y.F.CT.58319 (ct3) 

acc1x1x1_2 = [0.0250, 0.2250, 0.4500, 0.6556, 0.7333, 0.7917, 0.8472, 0.8833, 0.9028,  0.9222, 0.9278, 0.9333, 0.9333, 0.9333, 0.9444, 0.9528, 0.9722, 0.9750, 1.0000]
acc21x21x1_2 = [0.0917, 0.4917, 0.6778, 0.7944, 0.8889, 0.9083, 0.9333, 0.9389, 0.9417,  0.9556, 0.9639, 0.9722, 0.9722, 0.9722, 0.9778,  0.9917, 1.0000, 1.0000, 1.0000]
acc1x1x5_2 = [0.0583, 0.4778, 0.7278, 0.8667, 0.9222, 0.9472, 0.9472, 0.9500, 0.9500, 0.9583, .9667, 0.9750, 0.9750, 0.9750, 0.9750, 0.9806,  0.9861, 0.9944, 1.0000] 
acc21x21x5_2 = [0.2139, 0.8194, 0.9528, 0.9917, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
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

acc1x1x1_3 = [ 0.0167, 0.2167, 0.3889, 0.4972, 0.6000, 0.6833, 0.7389, 0.7639, 0.7750, 0.7917, 0.8028, 0.8194, 0.8278, 0.8528, 0.9000, 0.9222, 0.9389, 0.9833, 1.0000]
acc21x21x1_3 = [0.0556, 0.3889, 0.5667, 0.6917, 0.7583, 0.7861, 0.8139, 0.8306, 0.8444, 0.8500, 0.8556, 0.8750, 0.8944, 0.9278, 0.9639, 0.9806, 0.9917, 1.0000, 1.0000]
acc1x1x5_3 = [0.0833, 0.5250, 0.6833, 0.7611, 0.7944, 0.8083, 0.8250, 0.8472, 0.8583, 0.8639, 0.8778, 0.8972, 0.9111, 0.9306, 0.9611, 0.9722, 0.9944, 1.0000, 1.0000] 
acc21x21x5_3 = [0.1861, 0.7444, 0.8778, 0.9139, 0.9222, 0.9278, 0.9389, 0.9500, 0.9556, 0.9556, 0.9639, 0.9639, 0.9722, 0.9833, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.037Y.M.CT.57613")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_3, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_3, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_3, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_3, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.037Y.M.CT.57613.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

# SMIR.Body.040Y.M.CT.57768

acc1x1x1_4 = [0.0306, 0.2944,  0.5417, 0.6778, 0.7694, 0.8556, 0.9083, 0.9472, 0.9472, 0.9472, 0.9556, 0.9583, 0.9611, 0.9611, 0.9611, 0.9694, 0.9889, 0.9889, 1.0000]
acc21x21x1_4 = [0.1194, 0.5667, 0.7889, 0.8667, 0.9194, 0.9556, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9806, 0.9944, 1.0000, 1.0000]
acc1x1x5_4 = [0.0889, 0.5944, 0.8250, 0.9250, 0.9639, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9944, 0.9944, 0.9972, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
acc21x21x5_4 = [0.2306, 0.8778, 0.9694, 0.9861, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] 
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.040Y.M.CT.57768")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_4, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_4, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_4, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_4, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.040Y.M.CT.57768.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# ----------------------------------------------------------------
# SMIR.Body.041Y.F.CT.57699

acc1x1x1_5 = [0.0056, 0.0806, 0.1806, 0.2722, 0.3417, 0.4083, 0.5028, 0.6028, 0.6444, 0.6722, 0.7083, 0.7444, 0.7889, 0.8528, 0.8833, 0.9056, 0.9222, 0.9667, 1.0000]
acc21x21x1_5 = [0.0333, 0.2750, 0.3917, 0.4694, 0.5333, 0.6167, 0.6667, 0.7444, 0.7944, 0.8361, 0.8667, 0.8750, 0.8889, 0.9056, 0.9306, 0.9444, 0.9667, 1.0000, 1.0000]
acc1x1x5_5 = [0.0306, 0.3000, 0.5000, 0.5833, 0.6194, 0.6611, 0.7056, 0.7472, 0.7694, 0.8000, 0.8528, 0.8778, 0.8972, 0.9111, 0.9139, 0.9306, 0.9806, 0.9972, 1.0000] 
acc21x21x5_5 = [0.0972, 0.5278, 0.6944, 0.7667, 0.8250, 0.8722, 0.9000, 0.9250, 0.9389, 0.9472, 0.9472, 0.9556, 0.9611, 0.9667, 0.9750, 0.9806, 0.9861, 1.0000, 1.0000] 
plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.041Y.F.CT.57699")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_5, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_5, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_5, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_5, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.041Y.F.CT.57699.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# SMIR.Body.057Y.F.CT.59693

acc1x1x1_6 = [0.0139, 0.1583, 0.2694, 0.3889, 0.4639, 0.5194, 0.5639, 0.6250, 0.6806, 0.7278, 0.7889, 0.8250, 0.8694, 0.8722, 0.8861, 0.9139, 0.9306, 0.9583, 1.0000]
acc21x21x1_6 = [0.0361, 0.2333, 0.3861,  0.4889, 0.5611, 0.6083, 0.6750, 0.7306, 0.7694, 0.8111, 0.8472, 0.8722, 0.8861, 0.8889, 0.9056, 0.9194, 0.9556, 0.9972, 1.0000]
acc1x1x5_6 = [0.0389, 0.2778, 0.4722, 0.6583, 0.7056, 0.7444, 0.7667, 0.8083, 0.8333, 0.8472, 0.8750, 0.8861, 0.9028, 0.9417, 0.9722, 0.9778, 0.9944, 1.0000, 1.0000] 
acc21x21x5_6 = [0.1167, 0.4917, 0.7056, 0.8028, 0.8444, 0.8778, 0.8833, 0.8944, 0.9167, 0.9222, 0.9306, 0.9361, 0.9444, 0.9583, 0.9861, 1.0000, 1.0000, 1.0000, 1.0000] 

plt.figure(figsize=[8, 5])
plt.title("SMIR.Body.057Y.F.CT.59693")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1x1x1_6, label=r"Reference: [1x1x1]")
plt.plot(thresholds, acc21x21x1_6, label=r"Translation: [21x21x1]")
plt.plot(thresholds, acc1x1x5_6, label=r"Scaling: [1x1x5]")
plt.plot(thresholds, acc21x21x5_6, label=r"Translation + scaling: [21x21x5]")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_SMIR.Body.057Y.F.CT.59693.png")


#-------------------------------------------------------------------------
acc1x1x1 = np.mean(np.array([acc1x1x1_1, acc1x1x1_2, acc1x1x1_3, acc1x1x1_4, acc1x1x1_5, acc1x1x1_6]), axis=0)
acc21x21x1 = np.mean(np.array([acc21x21x1_1, acc21x21x1_2, acc21x21x1_3, acc21x21x1_4, acc21x21x1_5, acc21x21x1_6]), axis=0)
acc1x1x5 = np.mean(np.array([acc1x1x5_1, acc1x1x5_2, acc1x1x5_3, acc1x1x5_4, acc1x1x5_5, acc1x1x5_6]), axis=0)
acc21x21x5 = np.mean(np.array([acc21x21x5_1, acc21x21x5_2, acc21x21x5_3, acc21x21x5_4, acc21x21x5_5, acc21x21x5_6]), axis=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.figure(figsize=[8, 5])
plt.title("Accuracy of the CNN (Training CT: SMIR.Body.058Y.M.CT.59468)")
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
plt.savefig("accuracy_combined_ct3.png")

print("Done!")
