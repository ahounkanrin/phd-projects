import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]




acc_ref_200x200 = [0.04444444, 0.41805556, 0.609375,   0.6875, 0.73263889, 0.76006944, 0.78402778, 0.80104167, 0.81875,    0.83854167, 0.85243056, 0.875,
 					0.88819444, 0.90625,    0.92256944, 0.94548611, 0.95868056, 0.97569444, 1.]
acc_trans_scale_200x200 = [0.796875,   0.98506944, 0.996875,   0.99826389, 0.99826389, 0.99826389, 0.99826389, 0.99826389, 0.99826389, 0.99826389, 								0.99861111, 0.99861111, 0.99930556, 1., 1., 1.,  1.,  1.,  1.]

acc_ref_400x400 = [0.05486111, 0.49375,    0.67708333, 0.73784722, 0.76631944, 0.78819444, 0.80451389, 0.82118056, 0.83819444, 0.85,       0.86423611, 						0.88333333, 0.90138889, 0.91319444, 0.92569444, 0.94201389, 0.96284722, 0.97986111, 1.]

acc_trans_scale_400x400 = [0.79826389, 0.99583333, 0.99895833, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 								0.99930556, 0.99930556, 1., 1., 1., 1., 1., 1., 1.]


plt.figure(figsize=[8, 5])
#plt.title(r"Angle $\theta_z$ estimation: $200\times 200$ Vs $400 \times 400$ subimages")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_ref_200x200, label=r" $200\times 200$")
plt.plot(thresholds, acc_ref_400x400, label=r"$400\times 400$")

#plt.plot(thresholds, acc_trans_scale_200x200, label=r"Translation and scaling ($200\times 200$)")
#plt.plot(thresholds, acc_trans_scale_400x400, label=r"Translation and scaling ($400\times 400$)")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_theta_z_200x200_Vs_400x400.png")


print("Done!")
