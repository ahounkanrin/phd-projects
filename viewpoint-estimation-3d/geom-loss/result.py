import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_geom_loss = 8.0
acc_geom_loss = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]

medErr_geom_loss_inplane_rotation = 10.0
acc_geom_loss_inplane_rotation = [0.0000, 0.3139, 0.5000, 0.6389, 0.6972, 0.7194,  0.7361, 0.7500, 0.7528, 0.7611, 0.7778, 0.7917]

medErr_geom_loss_3d = 7.0
acc_geom_loss_3d = [0.0333, 0.4250, 0.6111, 0.6972, 0.7556, 0.7861, 0.7917, 0.7944, 0.7944, 0.7944, 0.7944, 0.8056 ]

medErr_geom_loss_3d_inplane_rotation = 8.0
acc_geom_loss_3d_inplane_rotation = [0.0306, 0.3500, 0.6028, 0.7444, 0.8250, 0.8472, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667]

medErr_geom_loss_3d_outofplane_rotation = 9.0
acc_geom_loss_3d_outofplane_rotation = [0.0250, 0.3111, 0.5250, 0.6111, 0.6611, 0.6833, 0.7306, 0.7500, 0.7583, 0.7694, 0.7889, 0.7972]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_geom_loss, label="2D")
plt.plot(thresholds, acc_geom_loss_3d, label="3D")
plt.plot(thresholds, acc_geom_loss_inplane_rotation, label="2D + in-plane rotation")
plt.plot(thresholds, acc_geom_loss_3d_inplane_rotation, label="3D + in-plane rotation")
plt.plot(thresholds, acc_geom_loss_3d_outofplane_rotation, label="3D + out-of-plane rotation")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")


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

print("Done!")
