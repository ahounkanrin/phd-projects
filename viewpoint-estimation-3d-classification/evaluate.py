import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
from utils import geom_cross_entropy, geom_cross_entropy_el, angular_distance
num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
#tf.config.set_soft_device_placement(True)

print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)
nclasses = 36
data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/azimuth_elevation/"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

translation_window = [i for i in range(-20, 21, 5)]

def crop_image(img):
    tx = 0#np.random.choice(translation_window)
    ty = 0#np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(x, y1, y2):
    x = tf.map_fn(crop_image, x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, tf.one_hot(y1, nclasses//2), tf.one_hot(y2, nclasses)



@tf.function
def test_step(images, labelsx, labelsz):
    predx, predz = model(images, training=False)
    return predx, predz

# Load dataset
def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y1_train = hf.get('y1_train')
    y2_train = hf.get('y2_train')
    x_test = hf.get('x_test')
    y1_test = hf.get('y1_test')
    y2_test = hf.get('y2_test')

    x_train = np.array(x_train)
    y1_train = np.array(y1_train).astype(int)
    y2_train = np.array(y2_train).astype(int)
    x_test = np.array(x_test)
    y1_test = np.array(y1_test).astype(int)
    y2_test = np.array(y2_test).astype(int)
    return x_train, y1_train, y2_train, x_test, y1_test, y2_test

x1, y1_el, y1_az, x2, y2_el, y2_az = read_dataset(data_dir+"dataset_thetaxy.h5")

x_train = [0] * len(x1)
x_test = [0] * len(x2)
y_train_el= y1_el//10
y_train_az= y1_az//10
y_test_el = y2_el//10
y_test_az = y2_az//10

assert max(y_test_az) == 35 and max(y_test_el) == 17

for i in tqdm(range(len(x2))):
    img = np.squeeze(x2[i])
    x_test[i] = np.stack([img, img, img], axis=-1)
x_test = np.asarray(x_test)
print("shape of x_test", x_test.shape)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_el, y_test_az)).batch(1)
test_data = test_data.map(lambda x,y1, y2: preprocess(x, y1, y2), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#print(test_data)

# Define model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs_el = tf.keras.layers.Dense(nclasses//2, activation="softmax")(x)
outputs_az = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=[outputs_el, outputs_az])



# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy_elevation = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy_elevation")
train_accuracy_azimuth = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy_azimuth")
test_accuracy_elevation = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy_elevation")
test_accuracy_azimuth = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy_azimuth")

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-classification-thetax_z/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

checkpoint.restore(manager.checkpoints[-1])
model.summary()

# Training loop
step = 0
preds_x = []
preds_z = []
gt_x = []
gt_z = []
for images, labelsx, labelsz in tqdm(test_data):
    tic = time.time()
    predx, predz = test_step(images, labelsx, labelsz)
    preds_x.append(10*np.argmax(predx)) # multiply  by the bin size to get angle between 0-360
    preds_z.append(10*np.argmax(predz))
    gt_x.append(10*np.argmax(labelsx))
    gt_z.append(10*np.argmax(labelsz))     

# preds_x = []
# preds_z = []
# gt_x = []
# gt_z = []
# for test_images, test_labelsx, test_labelsz in tqdm(train_data.map(preprocess, 
#                                         num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):                                  
#     predx, predz = test_step(test_images, test_labelsx, test_labelsz)
#     preds_x.append(10*np.argmax(predx)) # multiply  by the bin size to get angle between 0-360
#     preds_z.append(10*np.argmax(predz))
#     gt_x.append(10*test_labels[0, 0])
#     gt_z.append(10*test_labels[0, 2])

errors_x = [tf.squeeze(angular_distance(gt_x[i], preds_x[i])) for i in range(len(gt_x))]
errors_z = [tf.squeeze(angular_distance(gt_z[i], preds_z[i])) for i in range(len(gt_z))]
#print(errors_z)
thresholds = np.array([theta for theta in range(0, 95, 10)])

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors_x))))
with open("mederr.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(errors_x))), file=f)

acc_theta_x = []

for theta in thresholds:
    acc_bool_x = np.array([errors_x[i] <= theta  for i in range(len(errors_x))])
    accx = np.mean(acc_bool_x)
    acc_theta_x.append(accx)
    print("Accuracy at thetax = {} is: {:.4f}".format(theta//10, accx))

acc_theta_z = []

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors_z))))
with open("mederr.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(errors_z))), file=f)
for theta in thresholds:
    acc_bool_z = np.array([errors_z[i] <= theta  for i in range(len(errors_z))])
    accz = np.mean(acc_bool_z)
    acc_theta_z.append(accz)
    print("Accuracy at thetaz = {} is: {:.4f}".format(theta//10, accz))

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (bins of 10 degrees)")
plt.xticks(ticks=[i//10 for i in range(0, 95, 10)])
plt.yticks(ticks=[i/20 for i in range(21)])
plt.plot(thresholds//10, acc_theta_x, label=r"Elevation ($\theta_x$)")
#plt.plot(thresholds, acc_thetay, label=r"Rotation angle $\theta_y$")
plt.plot(thresholds//10, acc_theta_z, label=r"Azimuth ($\theta_z$)")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_bins.png")