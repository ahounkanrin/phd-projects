import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_geom_loss = 8.0
acc_geom_loss = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]

medErr_geom_loss_inplane_rotation = 10.0
acc_geom_loss_inplane_rotation = [0.0000, 0.3139, 0.5000, 0.6389, 0.6972, 0.7194,  0.7361, 0.7500, 0.7528, 0.7611, 0.7778, 0.7917]

medErr_geom_loss_3d = 7.0
acc_geom_loss_3d = [0.0389, 0.3528, 0.6167, 0.6917, 0.7056, 0.7361, 0.7500, 0.7722, 0.7833, 0.8028, 0.8278, 0.8611]

medErr_geom_loss_3d_inplane_rotation = 6.0
acc_geom_loss_3d_inplane_rotation = [0.0222, 0.4556, 0.6306, 0.7139, 0.7500, 0.7583, 0.7611, 0.7667, 0.7750, 0.8000, 0.8222, 0.8306]

medErr_geom_loss_3d_outofplane_rotation = 8.0
acc_geom_loss_3d_outofplane_rotation = [0.0167, 0.3667,  0.5583, 0.6889, 0.7500, 0.7972, 0.8194, 0.8278, 0.8361, 0.8583, 0.8861, 0.9111]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_geom_loss, label="2D")
plt.plot(thresholds, acc_geom_loss_3d, label="3D")
#plt.plot(thresholds, acc_geom_loss_inplane_rotation, label="2D + in-plane rotation")
plt.plot(thresholds, acc_geom_loss_3d_inplane_rotation, label="3D + in-plane rotation")
plt.plot(thresholds, acc_geom_loss_3d_outofplane_rotation, label="3D + out-of-plane rotation")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("acc_3d.png")

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
