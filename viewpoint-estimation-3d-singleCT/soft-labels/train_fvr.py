import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
import cv2 as cv
from multiprocessing import Pool, cpu_count
import random
from utils import soft_label_encoding, geodesic_distance
from tqdm import tqdm

random.seed(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

def rotation_matrix(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    r = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return r

def rotate_plane(plane, rotationMatrix):
	rotatedPlane = np.matmul(rotationMatrix, plane)
	return rotatedPlane

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val)/(max_val - min_val)
    img = 255 * img 
    img = 255 - img  # use 255 for background pixel values
    img = np.uint8(img)
    return img

# Load ct volume
INPUT_SIZE = (200, 200)
imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
imgpath2 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.025Y.M.CT.57697/SMIR.Body.025Y.M.CT.57697.nii"
N = 512 
print("INFO: loading CT volume...")
tic_load = time.time()
ctVolume = nib.load(imgpath1).get_fdata().astype(int)
ctVolume = np.squeeze(ctVolume)
voi = ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume

val_ctVolume = nib.load(imgpath2).get_fdata().astype(int)
val_ctVolume = np.squeeze(val_ctVolume)
val_voi = val_ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
toc_load = time.time()
print("Done after {:.2f} seconds.".format(toc_load - tic_load)) 

tic_fft = time.time()
voiShifted = np.fft.fftshift(voi)
voiFFT = np.fft.fftn(voiShifted)
voiFFTShifted = np.fft.fftshift(voiFFT)

val_voiShifted = np.fft.fftshift(val_voi)
val_voiFFT = np.fft.fftn(val_voiShifted)
val_voiFFTShifted = np.fft.fftshift(val_voiFFT)
toc_fft = time.time()
print("3D FFT computed in {:.2f} seconds.".format(toc_fft - tic_fft))


# Rotation and Interpolation of the projection slice from the 3D FFT volume
x_axis = np.linspace(-N/2+0.5, N/2-0.5, N)
y_axis = np.linspace(-N/2+0.5, N/2-0.5, N)
z_axis = np.linspace(-N/2+0.5, N/2-0.5, N)

train_interpolating_function = RegularGridInterpolator((x_axis, y_axis, z_axis), voiFFTShifted)
val_interpolating_function = RegularGridInterpolator((x_axis, y_axis, z_axis), val_voiFFTShifted)
projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
projectionPlane = np.reshape(projectionPlane, (N, N, 3, 1), order="F")

def render_train_view(viewpoint):
    theta = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    rotationMatrix = rotation_matrix(theta)
    projectionSlice = rotate_plane(projectionPlane, rotationMatrix)
    projectionSlice = np.squeeze(projectionSlice)
    projectionSliceInterpolated =  train_interpolating_function(projectionSlice)     
    projectionSliceIFFT = np.abs(np.fft.ifft2(projectionSliceInterpolated))
    img = np.fft.fftshift(projectionSliceIFFT)
    img = normalize(img)
    img = img[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = soft_label_encoding(theta)
    return img, label

def render_val_view(viewpoint):
    theta = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    rotationMatrix = rotation_matrix(theta)
    projectionSlice = rotate_plane(projectionPlane, rotationMatrix)
    projectionSlice = np.squeeze(projectionSlice)
    projectionSliceInterpolated =  val_interpolating_function(projectionSlice)     
    projectionSliceIFFT = np.abs(np.fft.ifft2(projectionSliceInterpolated))
    img = np.fft.fftshift(projectionSliceIFFT)
    img = normalize(img)
    img = img[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = soft_label_encoding(theta)
    return img, label


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
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=2000, 
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
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
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
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)


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
            train_batch = pool.map(render_train_view, train_viewpoints_batch)

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
            val_batch = pool.map(render_val_view, val_viewpoints_batch)
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
    
    