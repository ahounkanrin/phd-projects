import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
from utils import geodesic_distance, geom_cross_entropy, one_hot_encoding, angular_distance, angular_distance2

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=360, type=int, help="Batch size")
    #parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    #parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

args = get_arguments()


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y



# Load dataset
print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)

data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/test/"
data_dir2 = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/test2/SMIR.Body.033Y.M.CT.57766/"
test_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
shuffle = False,
image_size=INPUT_SIZE,
batch_size=args.batch_size)
test_data = test_data.map(lambda x,y: preprocess(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE) 

print("Done.")


baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
model.trainable = False
#model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=500, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/geom-loss/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)
checkpoint.restore(manager.checkpoints[-1]) 
#checkpoint.restore("/scratch/hnkmah001/phd-projects/viewpoint-estimation2/geom-loss/checkpoints/ckpt-5")
pred = []
gt = []

for test_images, test_labels in tqdm(test_data):
        pred.append(np.argmax(model(test_images), axis=-1)) 
        gt.append(test_labels.numpy())

pred = np.array(pred).flatten()
gt = np.array(gt).flatten()


#gt = [np.argmax(label) for label in y_test]

thresholds = [theta for theta in range(0, 95, 5)]
    
error = [angular_distance(gt[i], pred[i]).numpy() for i in range(len(gt))]

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(error))))
with open("accuracy_mederr.txt", "w+") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(error))), file=f)

acc_list2 = []
for theta in thresholds:
    acc_bool2 = np.array([error[i] <= theta  for i in range(len(error))])
    acc2 = np.mean(acc_bool2)
    acc_list2.append(acc2)
    print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2))
    with open("accuracy_mederr.txt", "a") as f:
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc2), file=f)
    
