import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 60, 5)]


medErr_geom_loss = 8.0
acc_geom_loss = [0.0000, 0.3806, 0.5556, 0.6500, 0.7167, 0.7361, 0.7528, 0.7639, 0.7833, 0.8167, 0.8167, 0.8278]

#medErr_geom_loss_3d = 6.0
acc_geom_loss_3d = [0.0194, 0.4417, 0.6750, 0.7583, 0.7833, 0.7917, 0.7944, 0.7944, 0.7944, 0.7944, 0.7944, 0.7972]

medErr_geom_loss_3d_inplane_rotation = 6.0
acc_geom_loss_3d_inplane_rotation = [ 0.0222, 0.4556, 0.6306, 0.7139, 0.7500, 0.7583, 0.7611, 0.7667, 0.7750, 0.8000, 0.8222, 0.8306]

plt.figure(figsize=[8, 5])
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")

plt.plot(thresholds, acc_geom_loss, label="2D input")
plt.plot(thresholds, acc_geom_loss_3d, label="3D input")
plt.plot(thresholds, acc_geom_loss_3d_inplane_rotation, label="3D + In-plane-rotation")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")


medErrs = [medErr_geom_loss, medErr_geom_loss_3d_inplane_rotation]

plt.figure(figsize=[4, 4])
#plt.title("Median Error")
plt.ylabel("Median Error (degrees)")
plt.bar(0, medErrs[0],  label="2D")
#plt.bar(1, medErrs[1],  label="3D")
plt.bar(1, medErrs[1],  label="3D + In-plane rotation")
plt.xticks([0, 1], [])
plt.legend(loc="upper right")
plt.savefig("mederr.png")

print("Done!")
