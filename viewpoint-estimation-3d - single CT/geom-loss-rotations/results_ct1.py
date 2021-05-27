import numpy as np
from matplotlib import pyplot as plt

ctids = ["Body.025Y.M.CT.57697", "Body.025Y.M.CT.59477", "Body.030Y.F.CT.59466", "Body.031Y.M.CT.59692", "Body.033Y.M.CT.57764", "Body.033Y.M.CT.57766", "Body.036Y.F.CT.58319", "Body.037Y.F.CT.57796", "Body.037Y.M.CT.57613", "Body.038Y.M.CT.61174", "Body.039Y.F.CT.58316", "Body.040Y.M.CT.57768", "Body.043Y.M.CT.58317", "Body.045Y.M.CT.59467",
            "Body.045Y.M.CT.59470", "Body.045Y.M.CT.59481", "Body.047Y.F.CT.57792", "Body.049Y.M.CT.57791", "Body.052Y.M.CT.57765", "Body.052Y.M.CT.59475", "Body.053Y.M.CT.58322", "Body.057Y.F.CT.57793", "Body.057Y.F.CT.59693", "Body.057Y.M.CT.57609", "Body.058Y.M.CT.57767", "Body.058Y.M.CT.59468"]

accs = [0.9667, 0.9222, 0.7833, 0.7583,  0.8611, 0.9444, 0.9139, 0.8528, 0.8306, 0.6222, 0.6278, 0.9056, 0.8028, 0.8278, 0.8361, 0.9444, 0.8083, 0.6417, 0.9667, 0.6694, 0.7806, 0.7472, 0.8222, 0.9694, 0.7278, 0.7417]

accs_ref = [0.4556, 0.4583, 0.5528, 0.2917, 0.5167, 0.6833,  0.5861, 0.3222, 0.4278, 0.1917, 0.2944, 0.4194, 0.4056, 0.3056, 0.2861, 0.4639, 0.5417, 0.2139, 0.7250, 0.4639, 0.3556, 0.3806, 0.3694, 0.4444, 0.2694, 0.3667]

width = 0.3
x = np.arange(len(accs))
fig, ax = plt.subplots(figsize=(30, 40))
bar1 = ax.bar(x-width/2, accs_ref, width, label="Reference")
bar2 = ax.bar(x-width/2, accs, width, label="21x21x5")
plt.bar(ctids, accs_ref)
plt.yticks(ticks=[i/10 for i in range(11)], size=20, weight="bold")
plt.xticks(rotation=90, size=20, weight="bold")
plt.title(r"Accuracy at $\theta = 10$", size=20, weight="bold")
ax.legend()
plt.savefig("ct1_results_ref2.png")
print("Done")



