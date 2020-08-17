import keras
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py


def save_dataset(x_train, y_train, x_test, y_test, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_test', data=y_test)
    hf.close()

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpus", type=int, default=0, help="Number of gpus")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--is_training", type=lambda x: bool(int(x)),  default=True, help="Whether training or predicting")
    return parser.parse_args()

args = get_argument()

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
#-----------------------------------------------------------------------------------------
DATA_DIR = "/scratch/hnkmah001/Datasets/ImageCLEF09/"
#DATA_DIR = "/home/anicet/Datasets/ImageCLEF/imageclef09/"
print("Loading dataset...")
x_train, y_train, x_test, y_test = read_dataset(DATA_DIR + 'imageclef.h5')
print("Dataset loaded.")

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)

with K.name_scope("Encoder"):
    img_input = Input(shape=(512, 512, 1))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(img_input)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(maxpool3)
    encoded = MaxPooling2D(pool_size=(2, 2))(conv4)

with K.name_scope("Decoder"):
    conv5 = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(encoded)
    upsample1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(upsample1)
    upsample2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(upsample2)
    upsample3 = UpSampling2D(size=(2,2))(conv7)
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(upsample3)
    upsample4 = UpSampling2D(size=(2,2))(conv8)
    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(upsample4)


encoder = keras.models.Model(img_input, encoded)
autoencoder = keras.models.Model(img_input, decoded)
autoencoder.summary()

if args.ngpus > 0:
    encoder = multi_gpu_model(encoder, gpus=args.ngpus)
    autoencoder = multi_gpu_model(autoencoder, gpus=args.ngpus)

autoencoder.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9), loss="mse")
weight_path = "./weights_sigmoid.h5"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs/", update_freq="batch")
stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=5)
checkpoint_callback = keras.callbacks.ModelCheckpoint(weight_path, save_best_only=True, verbose=1)


if args.is_training:
    autoencoder.fit(x=x_train, y=x_train, epochs = args.epochs, validation_data=(x_test, x_test),
                    batch_size=args.batch_size, callbacks=[tensorboard_callback, stopping_callback, checkpoint_callback])
    
    autoencoder.save("model.h5")

else:
    autoencoder.load_weights(weight_path) 
    result = autoencoder.evaluate(x=x_test, y=x_test, verbose=1)
    print("mse={}".format(result))
    decoded_test = autoencoder.predict(x=x_test, verbose=1)
    encoded_test = encoder.predict(x=x_test, verbose=1)
    encoded_train = encoder.predict(x=x_train, verbose=1)
    save_dataset(encoded_train, y_train, encoded_test, y_test, DATA_DIR+"imageclef_encoded_sigmoid.h5")
    n = 20
    decoded_test = decoded_test[:n, :, :, 0]
    encoded_test = encoded_test[:n, :, :, 0]
    test_img = x_test[:n, :, :, 0]
    plt.figure(figsize=(40, 6))
    
    for i in range(n):
        # display original
        plt.subplot(3, n, i+1)
        plt.imshow(np.squeeze(test_img[i]))
        plt.gray()

        # display encoded representations
        plt.subplot(3, n, i+1 + n)
        plt.imshow(np.squeeze(encoded_test[i]))
        plt.gray()

        # display reconstruction
        plt.subplot(3, n, i+1 + 2*n)
        plt.imshow(np.squeeze(decoded_test[i]))
        plt.gray()
        
    plt.show()
    plt.savefig("encodings_sigmoid.png")