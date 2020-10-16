import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
import argparse
from utils import get_view, rotation_matrix, soft_label_encoding, one_hot_encoding, geodesic_distance
from tqdm import tqdm
import time
import nibabel as nib
from scipy import ndimage
from multiprocessing import Pool, cpu_count



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

def get_train_views(viewpoint):
    theta = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    image = get_view(train_img3d, theta)
    img = image[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = soft_label_encoding(theta)
    return img, label

def get_val_views(viewpoint):
    theta = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    image = get_view(val_img3d, theta)
    img = image[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = soft_label_encoding(theta)
    return img, label


# Load data
INPUT_SIZE = (200, 200)
imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
imgpath2 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"

print("INFO: loading CT volumes...")
train_img3d = nib.load(imgpath1).get_fdata().astype(int)
train_img3d = np.squeeze(train_img3d)
train_img3d = train_img3d[:,:, :512]

val_img3d = nib.load(imgpath2).get_fdata().astype(int)
val_img3d = np.squeeze(val_img3d)
val_img3d = val_img3d[:,:, :512]
print("Done!")

x1 = [(theta, tx, 0) for theta in range(360) for tx in range(-20, 21)]
x2 = [(theta, 0, ty) for theta in range(360) for ty in range(-20, 21) if ty!=0]
xtrain = x1 + x2 
xval = [(theta, 0, 0) for theta in range(360)]


# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)
model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")


@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation 
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Calculate metrics
    train_accuracy.update_state(labels, predictions)
    train_loss.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    return predictions


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)


train_viewpoints = [i for i in range(360)]
random.shuffle(train_viewpoints)
train_labels = np.array([soft_label_encoding(i) for i in train_viewpoints])

step=0
for epoch in range(args.epochs):
    
    xtrain_epoch = xtrain.copy()
    random.shuffle(xtrain_epoch)
    while len(xtrain_epoch) > 0:
        tic = time.time()
        x_train = []
        y_train = []
        if len(xtrain_epoch) >= cpu_count():
            train_viewpoints_batch = random.sample(xtrain_epoch, cpu_count())
        else:
            train_viewpoints_batch = xtrain_epoch
        for example in train_viewpoints_batch:
            xtrain_epoch.remove(example)
        
        with Pool() as pool:
            train_batch = pool.map(get_train_views, train_viewpoints_batch)

        train_batch = np.array(train_batch)
        for i in range(len(train_batch)):
            x_train.append(train_batch[i, 0])
            y_train.append(train_batch[i, 1])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100).batch(args.batch_size)
        
        # Save logs with TensorBoard Summary
        if step == 0:
            train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/logs/train"
            val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/logs/val"
            
            train_summary_writer = tf.summary.create_file_writer(train_logdir)
            val_summary_writer = tf.summary.create_file_writer(val_logdir)
        
        for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            
            train_step(images, labels)
            step += 1
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=step)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                tf.summary.image("image", images, step=step, max_outputs=1)
            toc = time.time()
            print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t  ({:.2f} seconds/step)".format(step, 
                train_loss.result(), train_accuracy.result(), toc-tic))
            
            train_loss.reset_states()
            train_accuracy.reset_states()

    epoch +=1
    xval_epoch = xval.copy()
    x_val = []
    y_val = []
    while len(xval_epoch) > 0:
        if len(xval_epoch) >= cpu_count():
            val_viewpoints_batch = xval_epoch[:cpu_count()]
        else:
            val_viewpoints_batch = xval_epoch
        
        with Pool() as pool:
            val_batch = pool.map(get_val_views, val_viewpoints_batch)
        val_batch = np.array(val_batch)
        for i in range(len(val_batch)):
            x_val.append(val_batch[i, 0])
            y_val.append(val_batch[i, 1])
        
        for example in val_viewpoints_batch:
            xval_epoch.remove(example)

    x_val = np.array(x_val)
    y_val = np.array(y_val)        
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1)

    pred = []
    for val_images, val_labels in tqdm(val_data.map(preprocess, 
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
        pred_it = test_step(val_images, val_labels)
        pred.append(np.argmax(pred_it))

    gt = [np.argmax(label) for label in y_val]
    errors = [geodesic_distance([gt[i], pred[i]]) for i in range(len(gt))]
    val_acc = np.mean(np.array(np.array([error <= 30  for error in errors])))

    with val_summary_writer.as_default():
        tf.summary.scalar("val_accuracy", val_acc, step=epoch)
        tf.summary.image("val_images", val_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {} \t Validation Accuracy: {:.4f}, ckpt {}\n\n"
    print(template.format(epoch, val_acc, ckpt_path))
    
    