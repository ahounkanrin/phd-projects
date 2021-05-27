import pandas as pd
import random


random.seed(1001)
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

data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/large_fov_no_background/"
df = pd.read_csv("dataset.csv", sep=",")
impath = df["image"].tolist()
imlabels = df["label"].tolist()
data = [img_label  for img_label in zip(impath, imlabels)]

train_data = random.sample(data, int(3*len(impath)/4))
val_data = [img_label for img_label in data if img_label not in train_data]
train_data = np.array(train_data)
val_data = np.array(val_data)

df_train = pd.DataFrame()
df_val = pd.DataFrame()

df_train["impath"] = train_data[:, 0]
df_train["label"] = train_data[:, 1]
df_val["impath"] = val_data[:, 0]
df_val["label"] = val_data[:, 1]

print("[INFO]", len(train_data))
print("[INFO]", len(val_data))
df_train.to_csv("train_rz.csv", sep=",", index=False)
df_val.to_csv("val_rz.csv", sep=",", index=False)
