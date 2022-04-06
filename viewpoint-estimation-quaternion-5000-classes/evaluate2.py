import os
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["OPENBLAS_NUM_THREADS"] = "8" 
os.environ["MKL_NUM_THREADS"] = "8" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" 
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time
import pandas as pd
from utils import euler_to_quaternion, quaternionLoss, quaternion_angle
from matplotlib import pyplot as plt

# tf.config.threading.set_inter_op_parallelism_threads(8)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.set_soft_device_placement(False)

print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)
nclasses = 5000

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

translation_window = [i for i in range(-20, 21, 5)]

def load_img(imgpath):
    raw_img = tf.io.read_file(imgpath)
    img = tf.io.decode_png(raw_img, channels=3)
    return img 

def crop_image(img):
    tx = 0 #np.random.choice(translation_window)
    ty = 0 #np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(imgpath, labels):
    img = tf.map_fn(load_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.map_fn(crop_image, img)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32))
    return img, labels

# def test_step(images, labels):
#     predictions = model(images, training=False)
#     return tf.argmax(predictions)

# Load dataset
train_df = pd.read_csv("train1.csv", sep=",")
train_qw = train_df["qw"].astype(float)
train_qx = train_df["qx"].astype(float)
train_qy = train_df["qy"].astype(float)
train_qz = train_df["qz"].astype(float)

data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/quaternions_20000_classes2/test/"
test_df = pd.read_csv("test2.csv", sep=",")
img_path = test_df["image"].apply(lambda imgID: data_dir+imgID)
qw = test_df["qw"].astype(float)
qx = test_df["qx"].astype(float)
qy = test_df["qy"].astype(float)
qz = test_df["qz"].astype(float)
# = test_df["class"].astype(int)
gt_quaternions = tf.stack([qw, qx, qy, qz], axis=-1)
gt_quaternions = tf.cast(gt_quaternions, dtype=tf.float32)

img_path_list = tf.constant(np.array(img_path))
#labels_list = tf.constant(np.array(q_class))
test_data = tf.data.Dataset.from_tensor_slices((img_path_list, gt_quaternions)).batch(1) #, labels_list

print(test_data)


# Define model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation=None)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs) #outputy,

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-quaternions-5000-classes/checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1])
model.summary()

# Training loop
preds_list = []
gt = []

#counter = 0
for test_images, gt_quat in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):                                  
    preds = model(test_images, training=False)
    preds_list.append(tf.squeeze(tf.argmax(preds, axis=-1)).numpy())
    gt.append(tf.squeeze(gt_quat).numpy())
    #print(tf.squeeze(gt_quat).numpy())
    # counter += 1
    # if counter == 5:
    #     break
    #print()
# print("prediction", preds_list)
# print("gt", gt)

vals_qw = tf.constant([train_qw[i] for i in range(nclasses)], dtype=tf.float32)
vals_qx = tf.constant([train_qx[i] for i in range(nclasses)], dtype=tf.float32)
vals_qy = tf.constant([train_qy[i] for i in range(nclasses)], dtype=tf.float32)
vals_qz = tf.constant([train_qz[i] for i in range(nclasses)], dtype=tf.float32)
keys_tensor = tf.constant([i for i in range(nclasses)])

hashtable_qw = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qw), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qx = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qx), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qy = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qy), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)
hashtable_qz = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_qz), 
                                            default_value=tf.constant(0.0, dtype=tf.float32), name=None)

pred_tensor = tf.constant(preds_list)
#gt = tf.constant(gt)
pred_qw, pred_qx, pred_qy, pred_qz = hashtable_qw.lookup(pred_tensor), hashtable_qx.lookup(pred_tensor), hashtable_qy.lookup(pred_tensor), hashtable_qz.lookup(pred_tensor)
pred_quaternions = tf.stack([pred_qw, pred_qx, pred_qy, pred_qz], axis=-1)
# gt_qw, gt_qx, gt_qy, gt_qz = hashtable_qw.lookup(gt), hashtable_qx.lookup(gt), hashtable_qy.lookup(gt), hashtable_qz.lookup(gt)
# gt_quaternions = tf.stack([gt_qw, gt_qx, gt_qy, gt_qz], axis=-1)

errors = [quaternion_angle(gt[i], pred_quaternions[i]) for i in range(len(gt))]

thresholds = np.array([theta for theta in range(0, 95, 10)])

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
with open("result_train1_test2.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(errors))), file=f)

acc_theta = []

for theta in thresholds:
    acc_bool = np.array([errors[i] <= theta  for i in range(len(errors))])
    acc = np.mean(acc_bool)
    acc_theta.append(acc)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))
    with open("result_train1_test2.txt", "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc), file=f)

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (degrees)")
plt.xticks(ticks=[i for i in range(0, 95, 10)])
plt.yticks(ticks=[i/10 for i in range(21)])
plt.plot(thresholds, acc_theta)

# plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_train1_test2.png")
