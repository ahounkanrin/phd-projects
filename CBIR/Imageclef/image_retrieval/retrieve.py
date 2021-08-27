import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import random
import pandas as pd 
import h5py
import cv2 as cv
from skimage.feature import hog
from matplotlib import pyplot as plt

print("INFO: Processing dataset...")
INPUT_SIZE = (256, 256)

train_df = pd.read_csv("train_old.csv", sep=";")
test_df = pd.read_csv("test_old.csv", sep=";")
classes = sorted(list(set(train_df["irma_code"])))
nclasses = len(classes)
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'
DATA_DIR_HOG = '/scratch/hnkmah001/Datasets/ImageCLEF09/hog_features/'

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

def read_hog_features(hf5):
    hf = h5py.File(hf5,'r')
    return hf.get('hog_features')

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
checkpoint_dir = "/scratch/hnkmah001/phd-projects/CBIR/Imageclef/cnn-lsr/{}/checkpoints/".format(args.cnn)

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)
checkpoint.restore(manager.checkpoints[-1]) 
#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation2/geom-loss/checkpoints/ckpt-5")
# pred = []
# gt = []
fig = plt.figure(figsize=(20, 20))
img_indexes = random.sample([i for i in range(1733)], 6)
display_row_number = 0
for img_index in img_indexes:
    test_img = x_test[img_index]
    test_img = cv.resize(test_img, (128, 128), interpolation=cv.INTER_AREA)
    test_img_hog = hog(test_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    test_img = test_img/255.
    test_img = np.expand_dims(test_img, axis=0)
    #test_label = y_test[img_index]

    prob = model(test_img, training=False)   
    pred = np.argmax(prob)

    pred_class_hogs = read_hog_features(DATA_DIR_HOG + str(pred)+ ".h5")
    pred_class_df = pd.read_csv(DATA_DIR_HOG + str(pred)+ ".csv", sep=",")
    pred_class_paths = pred_class_df["image"].tolist()
    hog_distances = np.linalg.norm(pred_class_hogs - test_img_hog, axis=-1)
    #hog_distances = list(hog_distances)
    n = 5
    least_n = []
    for i in range(n):
        least_n.append(np.argmin(hog_distances))
        hog_distances[np.argmin(hog_distances)] = np.inf
    retrieved_paths_list = [pred_class_paths[index] for index in least_n]
    #retrieved_imgs = [cv.imread(path) for path in retrieved_paths_list]
    retrieved_imgs = []
    for path  in retrieved_paths_list:
        img = cv.imread(path, 0)
        img = cv.copyMakeBorder(img,top=(512-img.shape[0])//2,bottom=(512-img.shape[0])//2,
                    left=(512-img.shape[1])//2, right=(512-img.shape[1])//2,
                    borderType=cv.BORDER_CONSTANT, value=0)
        img = cv.resize(img, (128, 128))
        retrieved_imgs.append(img)
    display_imgs = [x_test[img_index]]
    for j in range(5):
        display_imgs.append(retrieved_imgs[j])
    #print("INFO", display_imgs[0])
    for k in range(6):
        fig.add_subplot(6, 6, k+1 + display_row_number*6)
        plt.axis("off")
        plt.imshow(display_imgs[k])
        plt.gray()
    display_row_number += 1
    # print("Test image index:", img_index)
    # print("Closest training image indexes:", least_n)
plt.savefig("retrieved_imgs.png")
print("Images retrieved.")
# decod_dict = dict((x, y) for x,y in enumerate(classes))
# pred_codes = [decod_dict[x] for x in pred]

# df = pd.DataFrame()
# df["image_id"] = test_df["image_id"]
# df["irma_code"] = test_df["irma_code"]
# #df["label"] = labels
# df["prediction"] = pred_codes
# df.to_csv("predictions.csv", sep=",", index=False)
# print("Accuracy = {}".format(test_accuracy.result()))
# test_accuracy.reset_states()


    
