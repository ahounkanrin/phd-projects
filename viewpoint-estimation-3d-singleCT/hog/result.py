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

medErr_geom_loss_3d = 7.0
acc_geom_loss_3d = [0.0389, 0.3528, 0.6167, 0.6917, 0.7056, 0.7361, 0.7500, 0.7722, 0.7833, 0.8028, 0.8278, 0.8611]


medErr_hog16 = 8.0000
acc_hog16 = [0.0167, 0.3500, 0.5750, 0.7361, 0.8111, 0.8500, 0.8833, 0.8944, 0.8972, 0.9000, 0.9083, 0.9194]

medErr_hog32 = 11.0000
acc_hog32 = [0.0194, 0.2389, 0.4750, 0.6667, 0.7528, 0.7833, 0.8167, 0.8500, 0.8722, 0.8944, 0.8944, 0.9056]

medErr_hog64 = 15.0000
acc_hog64 = [0.0083, 0.1861, 0.3306, 0.5083, 0.6389, 0.7056, 0.7417, 0.7694, 0.8111, 0.8361, 0.8417, 0.8472]


medErr_geom_loss_3d_outofplane_rotation = 8.0
acc_geom_loss_3d_outofplane_rotation = [0.0167, 0.3667,  0.5583, 0.6889, 0.7500, 0.7972, 0.8194, 0.8278, 0.8361, 0.8583, 0.8861, 0.9111]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_random, label="Random classifier")
plt.plot(thresholds, acc_geom_loss_3d_outofplane_rotation, label="CNN")
plt.plot(thresholds, acc_hog64, label=r"MLP with HOG - $64 \times 64$")
plt.plot(thresholds, acc_hog32, label=r"MLP with HOG - $32 \times 32$")
plt.plot(thresholds, acc_hog16, label=r"MLP with HOG - $16 \times 16$")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_hog.png")

"""
medErrs = [medErr_geom_loss, medErr_geom_loss_3d_inplane_rotation, medErr_geom_loss_3d_outofplane_rotation]

plt.figure(figsize=[4, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="2D")
#plt.bar(1, medErrs[1],  label="3D")
plt.bar(1, medErrs[1],  label="3D + in-plane rotation")
plt.bar(2, medErrs[2], label="3D + out-of-plane rotation")
plt.xticks([0, 1, 2], [])
plt.legend(loc="upper right")
plt.savefig("mederr.png")
"""
print("Done!")
