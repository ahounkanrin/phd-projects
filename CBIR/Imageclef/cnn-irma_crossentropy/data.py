from os import sep
import pandas as pd
import numpy as np

train_df = pd.read_csv("train_old.csv", sep=";")
test_df = pd.read_csv("test_old.csv", sep=";")
print(test_df[:10])
train_df2 = pd.DataFrame()
train_df2["image"] = train_df["image_id"].apply(lambda x : "/scratch/hnkmah001/Datasets/ImageCLEF09/train/" + str(x) + ".png")
train_df2["irma_code"] = train_df["irma_code"]
test_df2 = pd.DataFrame()
test_df2["image"] = test_df["image_id"].apply(lambda x : "/scratch/hnkmah001/Datasets/ImageCLEF09/test/" + str(x) + ".png")
test_df2["irma_code"] = test_df["irma_code"]

test_df2.to_csv("test.csv", sep=";", index=False)
train_df2.to_csv("train.csv", sep=";", index=False)
print(sorted(list(set(train_df2["irma_code"]))))

# 169 classes in test data
# 193 classes in train data