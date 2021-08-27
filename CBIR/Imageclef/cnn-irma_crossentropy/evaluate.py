import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import os
import time
import pandas as pd 
import h5py
import cv2 as cv

print("INFO: Processing dataset...")
INPUT_SIZE = (256, 256)

train_df = pd.read_csv("train_old.csv", sep=";")
test_df = pd.read_csv("test_old.csv", sep=";")
classes = sorted(list(set(train_df["irma_code"])))
nclasses = len(classes)
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument("--cnn", choices=["densenet", "inception", "resenet", "vgg"], required=True, help="Model architecture")
    return parser.parse_args()

args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, tf.one_hot(y, nclasses)

@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(labels, predictions)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    test_loss.update_state(labels, predictions)
    test_accuracy.update_state(labels, predictions)
    return predictions

# Load dataset
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

x_test = [0] * len(x2)
y_test = y2

# for i in tqdm(range(len(x1))):
#     img = np.squeeze(x1[i])
#     #img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
#     clahe = cv.createCLAHE(clipLimit=2.55, tileGridSize=(8,8))
#     img_r = img
#     img_g = clahe.apply(img)
#     img_b = cv.fastNlMeansDenoising(img, h=2, templateWindowSize=4, searchWindowSize=4)
#     x_train[i] = np.stack([img_r, img_g, img_b], axis=-1)
# x_train = np.asarray(x_train)
# print("shape of x_train:", x_train.shape)

clahe = cv.createCLAHE(clipLimit=2.55, tileGridSize=(8,8))
for i in tqdm(range(len(x2))):
    img = np.squeeze(x2[i])
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    # img_r = img
    # img_g = clahe.apply(img)
    # img_b = cv.fastNlMeansDenoising(img, h=2, templateWindowSize=4, searchWindowSize=4)
    x_test[i] = np.stack([img, img, img], axis=-1)
x_test = np.asarray(x_test)
print("shape of x_test", x_test.shape)

# x_train = tf.constant(x_train / 255.0, dtype=tf.float32)
# x_test = tf.constant(x_test / 255.0, dtype=tf.float32)

# all_train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
# train_data = all_train_data.take(int(0.8*len(x_train))).batch(args.batch_size)
# val_data = all_train_data.skip(int(0.8*len(x_train))).batch(args.batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

if args.cnn == "densenet":
    baseModel = tf.keras.applications.DenseNet121(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "inception":
    baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "resenet":
    baseModel = tf.keras.applications.ResNet101(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "vgg":
    baseModel = tf.keras.applications.VGG16(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")

inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy()
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
test_loss = tf.keras.metrics.CategoricalCrossentropy(name="test_loss")

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/CBIR/Imageclef/cnn-irma_crossentropy/{}/checkpoints/".format(args.cnn)

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)
checkpoint.restore(manager.checkpoints[-1]) 
#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation2/geom-loss/checkpoints/ckpt-5")
pred = []
gt = []

#step = 0
for test_images, test_labels in tqdm(test_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)):
    prob = test_step(test_images, test_labels)   
    pred.append(np.argmax(prob))
    gt.append(np.argmax(test_labels))

decod_dict = dict((x, y) for x,y in enumerate(classes))
pred_codes = [decod_dict[x] for x in pred]
#labels = [decod_dict[x] for x in gt]

df = pd.DataFrame()
df["image_id"] = test_df["image_id"]
df["irma_code"] = test_df["irma_code"]
#df["label"] = labels
df["prediction"] = pred_codes
df.to_csv("predictions.csv", sep=",", index=False)
print("Accuracy = {}".format(test_accuracy.result()))
test_accuracy.reset_states()


    
