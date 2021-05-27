import keras
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import regularizers
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from random import sample

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpus", type=int, default=0, help="Number of gpus")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--is_training", type=lambda x: bool(int(x)),  default=False, help="Whether training or predicting")
    return parser.parse_args()


def weighted_sse(x_true, x_pred):
    x_true = K.flatten(x_true)
    x_pred = K.flatten(x_pred)
    loss = x_true * K.square(x_true-x_pred)
    return K.mean(loss)

args = get_argument()

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

if args.ngpus > 1:
    encoder = multi_gpu_model(encoder, gpus=args.ngpus)
    autoencoder = multi_gpu_model(autoencoder, gpus=args.ngpus)

autoencoder.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9), loss="mse")
weight_path = "/scratch/hnkmah001/feature-extraction/checkpoints/weights_400x400.hdf5"
log_dir="/scratch/hnkmah001/feature-extraction/logs/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")
stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=3)
checkpoint_callback = keras.callbacks.ModelCheckpoint(weight_path, save_best_only=True, verbose=1)


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

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(dataframe=train_df, x_col="impath", class_mode="input", 
                                              target_size=(400, 400), color_mode="grayscale", 
                                              batch_size=args.batch_size, shuffle=True) 
                                              #save_to_dir="/scratch/hnkmah001/feature-extraction/train/")
validation_generator = datagen.flow_from_dataframe(dataframe=val_df, x_col="impath", class_mode="input", 
                                                   target_size=(400, 400), color_mode="grayscale", 
                                                   batch_size=1, shuffle=False)
                                                   #save_to_dir="/scratch/hnkmah001/feature-extraction/val/")
autoencoder.summary()
if args.is_training:
    autoencoder.fit_generator(train_generator, epochs = args.epochs, validation_data=validation_generator,
                              callbacks=[stopping_callback, checkpoint_callback, tensorboard_callback])
    
    result = autoencoder.evaluate_generator(validation_generator, verbose=1)
    print("mse={}".format(result))


else:
    #autoencoder.summary()
    autoencoder.load_weights(weight_path) 
    result = autoencoder.evaluate_generator(validation_generator, verbose=1)
    print("mse={}".format(result))
    decoded_imgs = autoencoder.predict_generator(generator=validation_generator, verbose=1)
    encoded_imgs = encoder.predict_generator(generator=validation_generator, verbose=1)
    n = 20
    decoded_imgs = decoded_imgs[:n, :, :, 0]
    encoded_imgs = encoded_imgs[:n, :, :, 0]
    val_img_list = val_df["impath"].tolist()[:n]
    plt.figure(figsize=(60, 8))
    
    for i in range(n):
        # display original
        plt.subplot(3, n, i+1)
        img = cv.imread(val_img_list[i], 0)
        #img = cv.resize(img, dsize=(128, 128), interpolation=cv.INTER_NEAREST)
        plt.imshow(img)
        plt.gray()

        # display feature map
        plt.subplot(3, n, i+1 + n)
        plt.imshow(np.squeeze(encoded_imgs[i]))
        plt.gray()

        # display reconstruction
        plt.subplot(3, n, i+1 + 2*n)
        plt.imshow(np.squeeze(decoded_imgs[i]))
        plt.gray()
        
    plt.show()
    plt.savefig("encodings.png")
