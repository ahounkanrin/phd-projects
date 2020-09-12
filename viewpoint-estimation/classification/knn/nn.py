import keras
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import pandas as pd
import cv2 as cv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from statistics import mode
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from utils import geodesic_distance, rotation_matrix


with K.name_scope("Encoder"):
    img_input = Input(shape=(400, 400, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(img_input)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(maxpool3)
    encoded = MaxPooling2D(pool_size=(2, 2))(conv4)

with K.name_scope("Decoder"):
    deconv1 = Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(encoded)
    upsample1 = UpSampling2D(size=(2, 2))(deconv1)
    deconv2 = Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(upsample1)
    upsample2 = UpSampling2D(size=(2,2))(deconv2)
    deconv3 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(upsample2)
    upsample3 = UpSampling2D(size=(2,2))(deconv3)
    deconv4 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(upsample3)
    upsample4 = UpSampling2D(size=(2,2))(deconv4)
    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(upsample4)


encoder = keras.models.Model(img_input, encoded)
autoencoder = keras.models.Model(img_input, decoded)
encoder = multi_gpu_model(encoder, gpus=2)
autoencoder = multi_gpu_model(autoencoder, gpus=2)
autoencoder.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9), loss="mse")
weight_path = "/scratch/hnkmah001/feature-extraction/checkpoints/weights_400x400.hdf5"

autoencoder.load_weights(weight_path)
autoencoder.summary()
k = 1

def preprocess(img):
    img = cv.resize(img, dsize=(400, 400))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1) 
    return img/255.   

def features(imgpath):
    img = cv.imread(imgpath, 0)
    img = preprocess(img)
    img_feature = encoder.predict(img)
    return np.squeeze(img_feature).flatten()

def most_common(lst):
    return max(set(lst), key=lst.count)

DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"
def fullpath(path):
    return DIR+path

traindf = pd.read_csv("train_rz.csv", sep=",")
valdf = pd.read_csv("val_rz.csv", sep=",")
testdf = pd.read_csv("test_rz.csv", sep=",")

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df["impath"] = traindf["impath"].apply(fullpath)
train_df["label"] = traindf["label"].astype(str)
val_df["impath"] = valdf["impath"].apply(fullpath)
val_df["label"] = valdf["label"].astype(str)
test_df["impath"] = testdf["impath"].apply(fullpath)
test_df["label"] = testdf["label"].astype(str)





train_img = train_df["impath"].tolist()
train_labels = train_df["label"].tolist()
train_label = [int(i) for i in train_labels]
test_img = test_df["impath"].tolist()
test_labels = test_df["label"].tolist()
test_label = [int(i) for i in test_labels]

train_features = [features(train_img[i]) for i in tqdm(range(len(train_img)), desc="Training")]

pred = []
for i in tqdm(range(len(test_img)), desc="Testing"):
    test_img_feacture = features(test_img[i])
    distances = [np.linalg.norm(test_img_feacture - ref_feature) for ref_feature in train_features]
    sorted_dist_indices = np.argsort(distances)
    knearest_dist_indices = sorted_dist_indices[:k] # Get indices of the k nearest features to test_img_feature
    knearest_labels = [train_label[j] for j in knearest_dist_indices]
    pred.append(most_common(knearest_labels))

gt = np.array(test_label)
pred = np.array(pred)

error2 = [geodesic_distance(rotation_matrix(gt[i]), rotation_matrix(pred[i])) for i in range(len(gt))]
    
print("\n\nMedian Error = {:.4f}".format(np.median(np.array(error2))))
with open("regression.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(error2))), file=f)

acc_list2 = []
thresholds = [theta for theta in range(0, 60, 5)]
for theta in thresholds:
    acc_bool2 = np.array([error2[i] <= theta  for i in range(len(error2))])
    acc2 = np.mean(acc_bool2)
    acc_list2.append(acc2)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2))
    with open("regression.txt", "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2), file=f)
    