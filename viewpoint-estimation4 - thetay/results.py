import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 61, 2)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]


acc1_1 = [0.09836065573770492, 0.39344262295081966, 0.6557377049180327, 0.7704918032786885, 0.9180327868852459, 0.9672131147540983, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0]
acc1_2 = [0.9180327868852459, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc2_1 = [0.01639344262295082, 0.16393442622950818, 0.36065573770491804, 0.47540983606557374, 0.5245901639344263, 0.5737704918032787, 
            0.6229508196721312, 0.6229508196721312, 0.639344262295082, 0.6721311475409836, 0.6721311475409836, 0.6721311475409836, 
            0.6885245901639344, 0.7049180327868853, 0.7377049180327869, 0.7377049180327869, 0.7868852459016393, 0.8032786885245902, 
            0.8032786885245902, 0.8032786885245902, 0.8032786885245902, 0.819672131147541, 0.8360655737704918, 0.9016393442622951, 
            0.9344262295081968, 0.9344262295081968, 0.9672131147540983, 1.0, 1.0, 1.0, 1.0]
acc2_2 = [0.8852459016393442, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc3_1 = [0.04918032786885246, 0.2459016393442623, 0.45901639344262296, 0.5573770491803278, 0.6229508196721312, 0.6721311475409836, 
            0.7213114754098361, 0.7704918032786885, 0.7868852459016393, 0.8032786885245902, 0.8032786885245902, 0.8032786885245902, 
            0.8032786885245902, 0.8360655737704918, 0.8852459016393442, 0.9180327868852459, 0.9508196721311475, 0.9836065573770492, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc3_2 = [0.7049180327868853, 0.8852459016393442, 0.9672131147540983, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 
            0.9836065573770492, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0]

acc4_1 = [0.03278688524590164, 0.18032786885245902, 0.3442622950819672, 0.5245901639344263, 0.5901639344262295, 0.6885245901639344, 
            0.8032786885245902, 0.8688524590163934, 0.9016393442622951, 0.9180327868852459, 0.9180327868852459, 0.9180327868852459, 
            0.9672131147540983, 0.9836065573770492, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc4_2 =  [0.9180327868852459, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc5_1 = [0.11475409836065574, 0.47540983606557374, 0.7377049180327869, 0.9180327868852459, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
acc5_2 = [0.9836065573770492, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc6_1 = [0.08196721311475409, 0.5573770491803278, 0.7868852459016393, 0.9180327868852459, 0.9672131147540983, 0.9836065573770492, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc6_2 = [0.9672131147540983, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc7_1 = [0.11475409836065574, 0.5409836065573771, 0.8032786885245902, 0.9344262295081968, 0.9836065573770492, 0.9836065573770492, 
            0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc7_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc8_1 = [0.01639344262295082, 0.08196721311475409, 0.36065573770491804, 0.45901639344262296, 0.639344262295082, 0.7704918032786885, 
            0.8524590163934426, 0.9016393442622951, 0.9344262295081968, 0.9672131147540983, 0.9836065573770492, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc8_2 = [0.9016393442622951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


acc_ref = np.mean([acc1_1, acc2_1, acc3_1, acc4_1, acc5_1, acc6_1, acc7_1, acc8_1], axis=0)
print(acc_ref)
acc_translation_scaling = np.mean([acc1_2, acc2_2, acc3_2, acc4_2, acc5_2, acc6_2, acc7_2, acc8_2], axis=0)
print(acc_translation_scaling)
plt.figure(figsize=[8, 5])
plt.title(r"Angle $\theta_x$ estiamtion (reference)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 61, 5)])
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

plt.savefig("accuracy_thetax_reference.png")

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
plt.title(r"Angle $\theta_x$ estimation (translation and scaling)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 61, 5)])
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

plt.savefig("accuracy_thetax_translation_scaling.png")


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
plt.title(r"Angle $\theta_x$ estimation (average)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 61, 5)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_ref, label="Reference")
plt.plot(thresholds, acc_translation_scaling, label="Translation and scaling")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_thetax_averages.png")

print("Done!")