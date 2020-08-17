import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import h5py


DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'

def padding(img):
    img = cv.copyMakeBorder(img,top=(512-img.shape[0])//2,bottom=(512-img.shape[0])//2,
                    left=(512-img.shape[1])//2, right=(512-img.shape[1])//2,
                    borderType=cv.BORDER_CONSTANT, value=0)
    img = cv.resize(img,(512,512))
    return img


def save_dataset(x_train, y_train, x_test, y_test, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_test', data=y_test)
    hf.close()


train_df = pd.read_csv(DATA_DIR+"train.csv", sep=";")
test_df = pd.read_csv(DATA_DIR+"test.csv", sep=";")
y_train = train_df["irma_code"].tolist()
y_test = test_df["irma_code"].tolist()
dt = dict([(y,x) for x,y in enumerate(sorted(set(y_train)))])
y_train = [dt[x] for x in y_train]
y_train = np.asarray(y_train, dtype=np.uint8)
y_test = [dt[x] for x in y_test]
y_test = np.asarray(y_test, dtype=np.uint8)

train_img_id = train_df["image_id"]
test_img_id = test_df["image_id"]
x_train = np.zeros([len(train_img_id), 512, 512, 1], dtype=np.uint8)
x_test = np.zeros([len(test_img_id), 512, 512, 1], dtype=np.uint8)

for i in tqdm(range(len(train_img_id))):
    img = cv.imread(DATA_DIR+"/train/"+str(train_img_id[i])+".png", 0)
    img = padding(img)
    x_train[i] = np.expand_dims(img, axis=-1)

for i in tqdm(range(len(test_img_id))):
    img = cv.imread(DATA_DIR+"/test/"+str(test_img_id[i])+".png", 0)
    img = padding(img)
    x_test[i] = np.expand_dims(img, axis=-1)



save_dataset(x_train, y_train, x_test, y_test, filename = DATA_DIR + 'imageclef.h5')