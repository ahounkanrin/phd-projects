import numpy as np
from matplotlib import pyplot as plt

print("INFO: Plotting...")
thresholds = [theta for theta in range(0, 95, 5)]

testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", "SMIR.Body.040Y.M.CT.57768", 
        "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]


acc1_1 = [0.05, 0.2972222222222222, 0.39166666666666666, 0.4444444444444444, 0.4666666666666667, 0.5194444444444445, 
        0.5722222222222222, 0.5833333333333334, 0.6416666666666667, 0.6722222222222223, 0.7055555555555556, 
        0.7444444444444445, 0.7666666666666667, 0.8083333333333333, 0.85, 0.9027777777777778, 0.9527777777777777, 
        0.9833333333333333, 1.0]
acc1_2 = [0.6861111111111111, 0.9944444444444445, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc2_1 = [0.04722222222222222, 0.48333333333333334, 0.6833333333333333, 0.7611111111111111, 0.7888888888888889, 0.8055555555555556, 
        0.8166666666666667, 0.8277777777777777, 0.85, 0.8638888888888889, 0.8777777777777778, 0.8888888888888888, 0.9111111111111111, 
        0.9222222222222223, 0.9416666666666667, 0.9611111111111111, 0.9805555555555555, 0.9861111111111112, 1.0]
acc2_2 = [0.7666666666666667, 0.9888888888888889, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc3_1 = [0.03611111111111111, 0.32222222222222224, 0.5444444444444444, 0.7027777777777777, 0.8, 0.85, 0.8805555555555555, 
         0.8805555555555555, 0.8944444444444445, 0.9222222222222223, 0.9361111111111111, 0.95, 0.9666666666666667, 
         0.9694444444444444, 0.9805555555555555, 0.9888888888888889, 0.9944444444444445, 1.0, 1.0]
acc3_2 = [0.5194444444444445, 0.9111111111111111, 0.95, 0.9611111111111111, 0.9666666666666667, 0.9722222222222222, 
        0.9833333333333333, 0.9888888888888889, 0.9916666666666667, 0.9916666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc4_1 = [0.019444444444444445, 0.18333333333333332, 0.3138888888888889, 0.36666666666666664, 0.4111111111111111, 0.4722222222222222, 
        0.5, 0.5555555555555556, 0.5861111111111111, 0.6166666666666667, 0.6416666666666667, 0.6916666666666667, 0.75, 
        0.7861111111111111, 0.8444444444444444, 0.8805555555555555, 0.9305555555555556, 0.975, 1.0]
acc4_2 = [0.6277777777777778, 0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 
        0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 0.9916666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

acc5_1 = [0.125, 0.8805555555555555, 0.9611111111111111, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889,
         0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9638888888888889, 0.9666666666666667, 0.9805555555555555, 
         0.9861111111111112, 0.9861111111111112, 0.9888888888888889, 0.9972222222222222, 1.0, 1.0]
acc5_2 = [0.9444444444444444, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc6_1 = [0.12222222222222222, 0.8833333333333333, 0.9888888888888889, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc6_2 = [0.9333333333333333, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

acc7_1 = [0.18055555555555555, 0.9777777777777777, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc7_2 = [0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

 
acc8_1 = [0.05555555555555555, 0.49166666666666664, 0.6611111111111111, 0.6888888888888889, 0.7138888888888889, 0.725, 
        0.7388888888888889, 0.7694444444444445, 0.7888888888888889, 0.8111111111111111, 0.8444444444444444, 0.8611111111111112, 
        0.8666666666666667, 0.8861111111111111, 0.9111111111111111, 0.9222222222222223, 0.9444444444444444, 0.9805555555555555, 1.0] 
acc8_2 = [0.8222222222222222, 0.9916666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 


acc_ref = np.mean([acc1_1, acc2_1, acc3_1, acc4_1, acc5_1, acc6_1, acc7_1, acc8_1], axis=0)
acc_translation_scaling = np.mean([acc1_2, acc2_2, acc3_2, acc4_2, acc5_2, acc6_2, acc7_2, acc8_2], axis=0)
plt.figure(figsize=[8, 5])
plt.title("Reference (central subimages)")
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

plt.savefig("accuracy_reference.png")

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
plt.title("Translation and scaling")
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

plt.savefig("accuracy_translation_scaling.png")


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
# plt.title("Translation and scaling")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_ref, label="Reference")
plt.plot(thresholds, acc_translation_scaling, label="Translation and scaling")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_averages.png")




acc_ref_no_aug = [0.05486111, 0.49375, 0.67708333, 0.73784722, 0.76631944, 0.78819444, 0.80451389, 0.82118056, 0.83819444, 0.85,
       0.86423611, 0.88333333, 0.90138889, 0.91319444, 0.92569444, 0.94201389, 0.96284722, 0.97986111, 1.]

acc_translation_scaling_no_aug = [0.79826389, 0.99583333, 0.99895833, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 0.99930556, 
        0.99930556, 0.99930556, 0.99930556, 0.99930556, 1.,   1., 1., 1., 1., 1., 1.]

acc_ref_inplane = [0.07951389, 0.56493056, 0.69305556, 0.74097222, 0.76805556, 0.79201389, 0.80902778, 0.82256944, 
        0.840625,   0.85625,    0.87118056, 0.88784722, 0.90520833, 0.91979167, 0.93923611, 0.95555556, 0.975, 0.990625, 1.]
acc_translation_scaling_inplane = [0.78125,   0.98472222, 0.99270833, 0.99409722, 0.99479167, 0.99548611, 
        0.996875,   0.99756944, 0.99791667, 0.99791667, 0.99895833, 1.,  1., 1., 1., 1., 1., 1., 1. ]

plt.figure(figsize=[8, 5])
plt.title("In-plane rotation data augmentation (reference)")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_ref_no_aug, label="Without rotation augmentation")
plt.plot(thresholds, acc_ref_inplane, label="With in-plane rotation")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_inplaneVsno-rotation-aug-during-training_ref.png")

plt.figure(figsize=[8, 5])
plt.title("In-plane rotation data augmentation")
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(11)])
plt.plot(thresholds, acc_translation_scaling_no_aug, label="Without rotation augmentation")
plt.plot(thresholds, acc_translation_scaling_inplane, label="With in-plane rotation")

plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("accuracy_inplaneVsno-rotation-aug-during-training.png")
print("Done!")