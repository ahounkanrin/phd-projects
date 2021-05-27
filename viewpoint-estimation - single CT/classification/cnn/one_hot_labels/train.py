import tensorflow as tf
import keras 
import numpy as np
import pandas as pd 
from datetime import datetime
import argparse
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics
import cv2 as cv
from utils import geodesic_distance, rotation_matrix

BATCH_SIZE = 16
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpus", type=int, default=1, help="Number of gpus")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--is_training", type=lambda x: bool(int(x)), default=True, help="Mode: training or testing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    return parser.parse_args()

args = get_arguments()

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

classes = train_df["label"].tolist()


logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq="batch")
weight_path = "./checkpoints/weights.h5"
model_path = "./checkpoints/model.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=weight_path, verbose=1, save_best_only=True)
stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="impath",
        y_col="label",
        target_size=(400, 400),
        color_mode="rgb",
        classes = classes,
        class_mode = "sparse",
        batch_size = args.ngpus * args.batch_size,
        interpolation="nearest")

#print("[INFO] - class indices", train_generator.class_indices)
val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="impath",
        y_col="label",
        target_size=(400, 400),
        color_mode="rgb",
        classes = classes,
        class_mode = "sparse",
        batch_size = args.ngpus * 4,
        shuffle=False,
        interpolation="nearest")

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="impath",
        y_col="label",
        target_size=(400, 400),
        color_mode="rgb",
        classes = classes,
        class_mode = "sparse",
        batch_size = args.ngpus * 4,
        shuffle=False,
        interpolation="nearest")

class_indices = train_generator.class_indices
decode_class_indices = {v: k for k, v in class_indices.items()}


baseModel = keras.applications.InceptionV3(input_shape=(400, 400, 3), include_top=False, weights="imagenet")
x = baseModel.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation="relu")(x)
predictions = keras.layers.Dense(360, activation="softmax")(x)

model = keras.Model(inputs=baseModel.input, outputs=predictions)

if args.ngpus > 1:
    training_model = multi_gpu_model(model, gpus=args.ngpus)

    training_model.summary()
else:
    training_model = model


training_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9),
                    metrics=['accuracy'])


training_model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    callbacks=[tensorboard_callback, checkpoint_callback, stopping_callback],
                    verbose=1,
                    epochs=args.epochs)
training_model.load_weights(weight_path)
model.save(model_path)
