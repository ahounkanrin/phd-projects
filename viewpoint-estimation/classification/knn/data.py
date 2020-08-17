import pandas as pd
import random


random.seed(9001)
"""
data_list = [i for i in range(360)]
train_list = random.sample(range(0, 360), 100)
val_list = sorted(train_list)
train_list = [ j for j in data_list if j not in val_list]

train_imgs = [str(i)+".png" for i in train_list]
val_imgs = [str(i)+".png" for i in val_list]

train_df = pd.DataFrame(data={"impath": train_imgs})
val_df = pd.DataFrame(data={"impath": val_imgs})
train_df.to_csv("train.csv", sep=",", index=False)
val_df.to_csv("val.csv", sep=",", index=False)
"""

#data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/"
data_dir = "/home/anicet/Datasets/ctfullbody/Person1/"
#data_split = ["rx/", "ry/", "rz/"]
data_split = ["rx/"]
img_path_list = []
for split in data_split:
    for i in range(1, 361):
        img_path_list.append(data_dir + split + str(i)+".png" )
df = pd.DataFrame(data={"impath": img_path_list})
df.to_csv("rx1.csv", sep=",", index=False) 


"""
df = pd.read_csv("data.csv")
impath = df["impath"].tolist()
impath = random.sample(impath, len(impath)) # Shuffle list of image paths
train_list = random.sample(impath, int(3*len(impath)/4))
val_list = [path for path in impath if path not in train_list]
df_train = pd.DataFrame(data={"impath": train_list})
df_val = pd.DataFrame(data={"impath": val_list})
print("[INFO]", len(train_list))
print("[INFO]", len(val_list))
df_train.to_csv("train.csv", sep=",", index=False)
df_val.to_csv("val.csv", sep=",", index=False)"""