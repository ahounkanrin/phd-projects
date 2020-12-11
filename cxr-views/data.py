import pandas as pd
import os

path = "/scratch/hnkmah001/Datasets/CXR-binary/"
filenames = []
for i in range(1, 16):
    f1 = next(os.walk(path+str(i)))[2]
    filenames = filenames + [str(i)+"/"+path for path in f1]

for i in range(76, 106):
    f2 = next(os.walk(path+str(i)))[2]
    filenames = filenames + [str(i)+"/"+ path for path in f2]

for i in range(166, 196):
    f3 = next(os.walk(path+str(i)))[2]
    filenames = filenames + [str(i)+"/"+path for path in f3]

for i in range(256, 286):
    f4 = next(os.walk(path+str(i)))[2]
    filenames = filenames + [str(i)+"/"+path for path in f4]

for i in range(346, 361):
    f5 = next(os.walk(path+str(i)))[2]
    filenames = filenames + [str(i)+"/"+path for path in f5]
print(len(filenames))
df = pd.DataFrame()
df["impath"] = filenames
df.to_csv("train.csv", index=False, sep=",")

