import numpy as np
import keras
from keras.utils import multi_gpu_model
import h5py
from datetime import datetime
import argparse
import math
from tqdm import tqdm
import pandas as pd
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3, MobileNet, MobileNetV2, Xception, DenseNet169, DenseNet201, DenseNet121
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
import cv2 as cv

num_classes = 193
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpus", type=int, default=0, help="Number of gpus")
    parser.add_argument("--augment", type=bool, default=False, help="Whether to apply data augmentation or not")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--is_training", type=(lambda x: bool(int(x))), default=1)
    return parser.parse_args()

args = get_arguments()
def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')
    x_test = hf.get('x_test')
    y_test = hf.get('y_test')

    x_train = np.array(x_train)
    y_train = np.array(y_train).astype(int)
    x_test = np.array(x_test)
    y_test = np.array(y_test).astype(int)
    return x_train, y_train, x_test, y_test

x1, y1, x2, y2 = read_dataset(DATA_DIR+"imageclef.h5")

def layering(img):
    img = np.repeat(img[..., np.newaxis], 3, -1)
    return img
x_train = [0] * len(x1)
x_test = [0] * len(x2)
y_train = y1
y_test = y2

for i in tqdm(range(len(x1))):
    img = x1[i]
    img = cv.resize(np.squeeze(img), (32, 32), interpolation=cv.INTER_NEAREST)
    x_train[i] = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
x_train = np.asarray(x_train)
print("shape of x_train:", x_train.shape)

for i in tqdm(range(len(x2))):
    img = x2[i]
    img = cv.resize(np.squeeze(img), (32, 32), interpolation=cv.INTER_NEAREST)
    x_test[i] = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
x_test = np.asarray(x_test)
print("shape of x_test", x_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

#base_model = VGG16(weights='imagenet', input_shape=(32,32,3), include_top=False)
base_model = DenseNet201(weights="imagenet", input_shape=(32, 32, 3), include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()

if args.ngpus > 0:
    model = multi_gpu_model(model, gpus=args.ngpus)


tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs/", write_images=True,update_freq="batch")
weight_path = "./weights.h5"
checkpoint_callback = ModelCheckpoint(filepath=weight_path, verbose=1, save_best_only=True)
stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9),
                metrics=['accuracy'])

if args.is_training:
    model.fit(x_train, y_train, verbose=1, batch_size=args.batch_size, epochs=args.epochs,validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback, checkpoint_callback, stopping_callback])
else:

    model.load_weights(weight_path)
    #model.load_weights("./densenet201/weights.h5") 
    prob = model.predict(x_test, verbose=1)
    pred = np.argmax(prob, axis=-1)
    accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=pred)
    print("Test accuracy: ", accuracy)
    sns.set(font_scale=3)
    print("Computing confusion matrix...")
    cm = sklearn.metrics.confusion_matrix(y_test, pred)
    print("Confusion matrix computed.")
    plt.figure(figsize=(40, 35))
    sns.heatmap(cm)
    plt.title("Accuracy={:1.4f}".format(accuracy), fontsize=40)
    plt.ylabel('Ground truth', fontsize=40)
    plt.xlabel('Predictions', fontsize=40)
    plt.savefig("confusion_matrix.png")

    train_df = pd.read_csv(DATA_DIR+"train.csv", sep=";")
    test_df = pd.read_csv(DATA_DIR+"test.csv", sep=";")
    labels = train_df["irma_code"].tolist()
    decode_dict = dict((x,y) for x,y in enumerate(sorted(set(labels))))
    pred_codes = [decode_dict[x] for x in pred]
    df = pd.DataFrame()
    df["image_id"] = test_df["image_id"]
    df["label"] = test_df["irma_code"]
    df["prediction"] = pred_codes
    df.to_csv("predictions.csv", sep=",")
    acc = sklearn.metrics.accuracy_score(y_true=test_df["irma_code"].tolist(), y_pred=pred_codes)
    print("Accuracy: ", acc)