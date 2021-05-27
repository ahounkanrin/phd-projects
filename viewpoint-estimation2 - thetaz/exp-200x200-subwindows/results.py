import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]


acc1_1 = acc =  [0.030555555555555555, 0.325, 0.5083333333333333, 0.6, 0.6361111111111111, 0.6444444444444445, 0.6694444444444444, 
                0.6861111111111111, 0.7027777777777777, 0.7277777777777777, 0.7555555555555555, 0.7861111111111111, 0.8111111111111111, 
                0.8444444444444444, 0.8694444444444445, 0.9166666666666666, 0.9444444444444444, 0.9638888888888889, 1.0] 
acc1_2 = [0.6916666666666667, 0.95, 0.9861111111111112, 0.9861111111111112, 0.9861111111111112, 0.9861111111111112, 0.9861111111111112, 
        0.9861111111111112, 0.9861111111111112, 0.9861111111111112, 0.9888888888888889, 0.9888888888888889, 0.9944444444444445, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0]

acc2_1 = [0.030555555555555555, 0.3194444444444444, 0.4888888888888889, 0.5972222222222222, 0.675, 0.6944444444444444, 0.7138888888888889, 
        0.7277777777777777, 0.7444444444444445, 0.7694444444444445, 0.7833333333333333, 0.8138888888888889, 0.8222222222222222, 0.8583333333333333, 
        0.8833333333333333, 0.9222222222222223, 0.9333333333333333, 0.9666666666666667, 1.0]
acc2_2 = [0.7111111111111111, 0.9444444444444444, 0.9944444444444445, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc3_1 = [0.030555555555555555, 0.28055555555555556, 0.40555555555555556, 0.5111111111111111, 0.6055555555555555, 0.6861111111111111, 0.7388888888888889, 
        0.7611111111111111, 0.7861111111111111, 0.825, 0.8444444444444444, 0.8611111111111112, 0.8805555555555555, 0.9027777777777778, 0.9361111111111111, 
        0.9666666666666667, 0.975, 0.9777777777777777, 1.0] 
acc3_2 = [0.6305555555555555, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc4_1 = [0.03611111111111111, 0.3333333333333333, 0.5388888888888889, 0.6111111111111112, 0.6277777777777778, 0.6472222222222223, 0.675, 0.6944444444444444, 
        0.7222222222222222, 0.7361111111111112, 0.7611111111111111, 0.7972222222222223, 0.825, 0.85, 0.8833333333333333, 0.9166666666666666, 0.9361111111111111, 
        0.9583333333333334, 1.0]
acc4_2 =  [0.7555555555555555, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc5_1 = [0.05555555555555555, 0.5833333333333334, 0.8305555555555556, 0.9027777777777778, 0.9166666666666666, 0.9388888888888889, 0.9444444444444444, 0.9583333333333334, 
        0.9583333333333334, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9694444444444444, 
        0.9805555555555555, 0.9916666666666667, 1.0]
acc5_2 = [0.9444444444444444, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc6_1 = [0.04722222222222222, 0.5194444444444445, 0.7583333333333333, 0.8333333333333334, 0.8722222222222222, 0.8944444444444445, 0.9055555555555556, 0.9083333333333333, 
        0.9194444444444444, 0.9333333333333333, 0.9388888888888889, 0.9444444444444444, 0.9444444444444444, 0.9527777777777777, 0.9583333333333334, 0.9666666666666667, 
        0.9722222222222222, 0.9888888888888889, 1.0]
acc6_2 = [0.9166666666666666, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc7_1 = [0.08611111111111111, 0.6138888888888889, 0.8194444444444444, 0.8611111111111112, 0.8833333333333333, 0.8972222222222223, 0.9083333333333333, 0.925, 0.9416666666666667, 
        0.9555555555555556, 0.9555555555555556, 0.9694444444444444, 0.9777777777777777, 0.9777777777777777, 0.9805555555555555, 0.9888888888888889, 
        0.9972222222222222, 0.9972222222222222, 1.0]
acc7_2 = [0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  

acc8_1 = [0.03888888888888889, 0.36944444444444446, 0.525, 0.5833333333333334, 0.6444444444444445, 0.6777777777777778, 0.7166666666666667, 0.7472222222222222, 0.775, 0.7972222222222223, 
        0.8166666666666667, 0.8638888888888889, 0.8805555555555555, 0.9, 0.9055555555555556, 0.9166666666666666, 0.9305555555555556, 0.9611111111111111, 1.0] 
acc8_2 = [0.775, 0.9861111111111112, 0.9944444444444445, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


acc_ref = np.mean([acc1_1, acc2_1, acc3_1, acc4_1, acc5_1, acc6_1, acc7_1, acc8_1], axis=0)
print(acc_ref)
acc_translation_scaling = np.mean([acc1_2, acc2_2, acc3_2, acc4_2, acc5_2, acc6_2, acc7_2, acc8_2], axis=0)
print(acc_translation_scaling)
plt.figure(figsize=[8, 5])
plt.title(r"Angle $\theta_z$ estimation: Reference ($200\times 200$ subimages)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1_1, label=testScans[0])
plt.plot(thresholds, acc2_1, label=testScans[1])
plt.plot(thresholds, acc3_1, label=testScans[2])
plt.plot(thresholds, acc4_1, label=testScans[3])
plt.plot(thresholds, acc5_1, label=testScans[4])
plt.plot(thresholds, acc6_1, label=testScans[5])
plt.plot(thresholds, acc7_1, label=testScans[6])
plt.plot(thresholds, acc8_1, label=testScans[7])

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_theta_z_reference_200x200.png")

# plt.figure(figsize=[8, 5])
# plt.title("Reference (central subimages)")
# plt.ylabel("Accuracy")
# plt.xlabel("Threshold (degrees)")
# plt.xticks(ticks=[i for i in range(0, 95, 10)])
# plt.yticks(ticks=[i/10 for i in range(11)])
# plt.plot(thresholds, acc_ref)
# #plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig("accuracy_average_reference.png")


plt.figure(figsize=[8, 5])
plt.title(r"Angle $\theta_z$ estimation: Translation and scaling($200\times 200$)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc1_2, label=testScans[0])
plt.plot(thresholds, acc2_2, label=testScans[1])
plt.plot(thresholds, acc3_2, label=testScans[2])
plt.plot(thresholds, acc4_2, label=testScans[3])
plt.plot(thresholds, acc5_2, label=testScans[4])
plt.plot(thresholds, acc6_2, label=testScans[5])
plt.plot(thresholds, acc7_2, label=testScans[6])
plt.plot(thresholds, acc8_2, label=testScans[7])

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_thetaz_translation_scaling_200x200.png")


# plt.figure(figsize=[8, 5])
# plt.title("Translation and scaling")
# plt.ylabel("Accuracy")
# plt.xlabel("Threshold (degrees)")
# plt.xticks(ticks=[i for i in range(0, 95, 10)])
# plt.yticks(ticks=[i/10 for i in range(11)])
# plt.plot(thresholds, acc_translation_scaling)
# #plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig("accuracy_average_translation_and_scaling.png")

plt.figure(figsize=[8, 5])
plt.title(r"Angle $\theta_z$ estimation: Averages ($200\times 200$)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_ref, label="Reference")
plt.plot(thresholds, acc_translation_scaling, label="Translation and scaling")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_thetaz_averages_200x200.png")

print("Done!")
