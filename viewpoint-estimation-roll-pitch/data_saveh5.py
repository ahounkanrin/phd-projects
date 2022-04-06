import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import h5py


DATA_DIR = '/scratch/hnkmah001/Datasets/ctfullbody/azimuth_elevation/'


def save_dataset(x_train, y1_train, y2_train, x_test, y1_test, y2_test, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('y1_train', data=y1_train)
    hf.create_dataset('y2_train', data=y2_train)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y1_test', data=y1_test)
    hf.create_dataset('y2_test', data=y2_test)
    hf.close()


train_df = pd.read_csv(DATA_DIR+"train.csv", sep=",")
test_df = pd.read_csv(DATA_DIR+"test.csv", sep=",")
y1_train = train_df["elevation"].tolist()
y2_train = train_df["azimuth"].tolist()
y1_test = test_df["elevation"].tolist()
y2_test = test_df["azimuth"].tolist() 
y1_train = np.asarray(y1_train, dtype=np.uint8)
y2_train = np.asarray(y2_train, dtype=np.uint8)
y1_test = np.asarray(y1_test, dtype=np.uint8)
y2_test = np.asarray(y2_test, dtype=np.uint8)

train_img_id = train_df["image"]
test_img_id = test_df["image"]
x_train = np.zeros([len(train_img_id), 512, 512, 1], dtype=np.uint8)
x_test = np.zeros([len(test_img_id), 512, 512, 1], dtype=np.uint8)

for i in tqdm(range(len(train_img_id))):
    img = cv.imread(DATA_DIR+"train-val/"+str(train_img_id[i]), 0)
    #img = padding(img)
    x_train[i] = np.expand_dims(img, axis=-1)

for i in tqdm(range(len(test_img_id))):
    img = cv.imread(DATA_DIR+"test/"+str(test_img_id[i]), 0)
    #img = padding(img)
    x_test[i] = np.expand_dims(img, axis=-1)



save_dataset(x_train, y1_train, y2_train, x_test, y1_test, y2_test, filename = DATA_DIR + 'dataset_thetaxy.h5')
