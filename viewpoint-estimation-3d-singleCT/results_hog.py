import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


# medErr_geom_loss = 8.0
# acc_geom_loss = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]

# medErr_geom_loss_inplane_rotation = 10.0
# acc_geom_loss_inplane_rotation = [0.0000, 0.3139, 0.5000, 0.6389, 0.6972, 0.7194,  0.7361, 0.7500, 0.7528, 0.7611, 0.7778, 0.7917]


medErr_random = 104.0000
acc_random = [0.0000, 0.0083, 0.0222, 0.0417, 0.0583, 0.0750, 0.1056, 0.1611, 0.2194, 0.2639,  0.2944, 0.3139]


medErr_hog8 = 10.0000
acc_hog8 = [0.0389, 0.3528, 0.5028, 0.5889, 0.6472, 0.7000, 0.7472, 0.7722, 0.8056, 0.8139, 0.8222, 0.8417]

medErr_hog16 = 8.0000
acc_hog16 = [0.0167, 0.3500, 0.5750, 0.7361, 0.8111, 0.8500, 0.8833, 0.8944, 0.8972, 0.9000, 0.9083, 0.9194]

medErr_hog32 = 11.0000
acc_hog32 = [0.0194, 0.2389, 0.4750, 0.6667, 0.7528, 0.7833, 0.8167, 0.8500, 0.8722, 0.8944, 0.8944, 0.9056]

medErr_hog64 = 15.0000
acc_hog64 = [0.0139, 0.2111, 0.3722, 0.5000, 0.6000, 0.6500, 0.6944, 0.7333, 0.7667, 0.7833, 0.7889, 0.7889]


#medErr_cnn = 5.5
#acc_cnn = [0.0167, 0.3667,  0.5583, 0.6889, 0.7500, 0.7972, 0.8194, 0.8278, 0.8361, 0.8583, 0.8861, 0.9111]


medErr_cnn2 = 8.0
acc_cnn2 = [0.0278, 0.4778, 0.6361, 0.7306, 0.7583, 0.7583, 0.7611, 0.7694, 0.7750, 0.7750, 0.7917, 0.8028]

medErr_hogPlusCNN = 5.0
acc_hogPlusCNN = [0.0278, 0.5306, 0.7389, 0.8333, 0.8944, 0.9139, 0.9306, 0.9389, 0.9417, 0.9472, 0.9500, 0.9500]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

#plt.plot(thresholds, acc_random, label="Random classifier", ls="--")
#plt.plot(thresholds, acc_cnn, label="CNN1", ls="--")
#plt.plot(thresholds, acc_cnn2, label="CNN2", ls="--")
plt.plot(thresholds, acc_cnn2, label="CNN", ls="--")
plt.plot(thresholds, acc_hog16, label="MLP-HOG")
plt.plot(thresholds, acc_hogPlusCNN, label="CNN+MLP-HOG")
#plt.plot(thresholds, acc_hog8, label=r"MLP with HOG - $8 \times 8$", color="k")
#plt.plot(thresholds, acc_hog16, label=r"MLP with HOG$")
#plt.plot(thresholds, acc_hog32, label=r"MLP with HOG - $32 \times 32$")
#plt.plot(thresholds, acc_hog64, label=r"MLP with HOG - $64 \times 64$")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_hogPlusCNN.png")


mean_acc = [np.mean(acc_cnn), np.mean(acc_cnn2), np.mean(acc_hog8), np.mean(acc_hog16), np.mean(acc_hog32), np.mean(acc_hog64)]


medErrs = [medErr_cnn, medErr_cnn2, medErr_hog8, medErr_hog16, medErr_hog32, medErr_hog64]

plt.figure(figsize=[6, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="CNN1")
plt.bar(1, medErrs[1],  label="CNN2")
plt.bar(2, medErrs[2], label="HOG8x8")
plt.bar(3, medErrs[3], label="HOG16x16")
plt.bar(4, medErrs[4], label="HOG32x32")
plt.bar(5, medErrs[5], label="HOG64x64")
plt.xticks([0, 1, 2, 3, 4, 5], [])
plt.legend(loc="lower left")
plt.savefig("mederr_hog.png")

plt.figure(figsize=[6, 4])
#plt.title("Median Error")
plt.ylabel("Mean Accuracy")
plt.bar(0, mean_acc[0],  label="CNN1")
plt.bar(1, mean_acc[1],  label="CNN2")
plt.bar(2, mean_acc[2], label="HOG8x8")
plt.bar(3, mean_acc[3], label="HOG16x16")
plt.bar(4, mean_acc[4], label="HOG32x32")
plt.bar(5, mean_acc[5], label="HOG64x64")
plt.xticks([0, 1, 2, 3, 4, 5], [])
plt.legend(loc="lower left")
plt.savefig("mean_acc_hog.png")
print("Done!")
