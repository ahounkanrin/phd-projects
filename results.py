import matplotlib.pyplot as plt
import numpy as np


thresholds = [theta for theta in range(0, 95, 5)]
acc_thetax = [0.11319444, 0.82951389, 0.94513889, 0.96979167, 0.98298611, 0.98680556, 0.98854167, 0.99201389, 
                0.99409722, 0.99479167, 0.99722222, 0.99861111, 1.,  1., 1., 1., 1., 1., 1.]

acc_thetay = acc =  [0.11458333, 0.84270833, 0.98333333, 0.99930556, 1.,  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] 

acc_thetaz = [0.08993056, 0.63090278, 0.75763889, 0.79131944, 0.809375, 0.82256944, 0.83611111, 0.84791667, 0.86701389, 0.88020833, 
            0.89375, 0.90833333, 0.92083333, 0.93715278, 0.95243056, 0.96597222, 0.975, 0.98888889, 1.]
plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_thetax, label=r"Elevation ($\theta_x$)")
plt.plot(thresholds, acc_thetay, label=r"In-plane rotation ($\theta_y$)")
plt.plot(thresholds, acc_thetaz, label=r"Azimuth ($\theta_z$)")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy.png")
