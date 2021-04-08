import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import time
from scipy.interpolate import interpn
from scipy.fft import fftn, fftshift, ifft2
import cv2 as cv
from multiprocessing import Pool, cpu_count
import random
from utils import one_hot_encoding, geodesic_distance, geom_cross_entropy
from tqdm import tqdm
from skimage.feature import hog

random.seed(0)
tf.random.set_seed(0)
min_ctnumber = -1024

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

#def preprocess(x, y):
#    x = tf.cast(x, dtype=tf.float32)
#    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
#    return x, y

def Rx(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    x = theta * np.pi / 180.
    r11, r12, r13 = np.cos(x), -np.sin(x), 0.
    r21, r22, r23 = np.sin(x), np.cos(x), 0.
    r31, r32, r33 = 0., 0., 1.
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def rotate_plane(plane, rotationMatrix):
	rotatedPlane = np.matmul(rotationMatrix, plane)
	return rotatedPlane

def normalize(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = 255 * img 
    return np.uint8(img)

#print("number of cpus", cpu_count())
# Load ct volume
INPUT_SIZE = (200, 200)
imgpath1 = "/scratch/hnkmah001/Datasets/ctfullbody/SMIR.Body.021Y.M.CT.57761/SMIR.Body.021Y.M.CT.57761.nii"
N = 512 
print("INFO: loading CT volume...")
ctVolume = nib.load(imgpath1).get_fdata().astype(int)
ctVolume = np.squeeze(ctVolume)
voi = ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
#voi = normalize(voi)    # Rescale CT numbers between 0 and 255
voi = voi - min_ctnumber
voi = np.pad(voi, N//2, "constant", constant_values=0)
print("Done.") 

voiShifted = np.fft.fftshift(voi)
voiFFT = np.fft.fftn(voiShifted)
voiFFTShifted = np.fft.fftshift(voiFFT)
print("3D FFT computed.")


# Rotation and Interpolation of the projection slice from the 3D FFT volume
x_axis = np.linspace(-N+0.5, N-0.5, 2*N)
y_axis = np.linspace(-N+0.5, N-0.5, 2*N)
z_axis = np.linspace(-N+0.5, N-0.5, 2*N)

projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

def render_train_view(viewpoint):
    theta_x = np.random.randint(-5, 5)
    theta_y = np.random.randint(-5, 5)
    theta_z = viewpoint[0]
    tx = viewpoint[1]
    ty = viewpoint[2]
    rotationMatrix = Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)
    projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=voiFFTShifted, xi=projectionSlice, method="linear",
                                    bounds_error=False, fill_value=0)      
    img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    img = img[N//2:N+N//2, N//2:N+N//2]
    img = normalize(img)
    img = img[54+tx:454+tx, 63+ty:463+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    hogFeature = hog(img, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), visualize=False, multichannel=None)
    label = one_hot_encoding(theta_z)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    return img, label, hogFeature


x1 = [(theta_z, tx, 0) for theta_z in range(360) for tx in range(-20, 21)]
x2 = [(theta_z, 0, ty) for theta_z in range(360) for ty in range(-20, 21) if ty!=0]
xtrainval = x1 + x2 
xval = random.sample(xtrainval, len(xtrainval)//4)
for val_example in xval:
        xtrainval.remove(val_example)
xtrain = xtrainval.copy()

xval_epoch = xval.copy()
x_val = []
y_val = []
hog_val = []
print("Generating validation data...")
while len(xval_epoch) > 0:
    if len(xval_epoch) >= args.batch_size:
        val_viewpoints_batch = random.sample(xval_epoch, args.batch_size)
    else:
        val_viewpoints_batch = xval_epoch.copy()
    for example in val_viewpoints_batch:
        xval_epoch.remove(example)

    with Pool() as pool:
        val_batch = pool.map(render_train_view, val_viewpoints_batch)
    val_batch = np.array(val_batch)
    for i in range(len(val_batch)):
        x_val.append(val_batch[i, 0])
        y_val.append(val_batch[i, 1])
        hog_val.append(val_batch[i, 2])
    
x_val = np.array(x_val)
y_val = np.array(y_val) 
hog_val = np.array(hog_val)       
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val, hog_val)).batch(args.batch_size)

# Define the model

baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
input_cnn = baseModel.input
input_hog = tf.keras.Input(shape=(4356,), name="HOG_features") 
features_cnn = tf.keras.layers.GlobalAveragePooling2D()(baseModel.output)
features_hog = tf.keras.layers.Flatten()(input_hog)
x = tf.keras.layers.Concatenate()([features_cnn, features_hog])
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)
model = tf.keras.Model(inputs=[input_cnn, input_hog], outputs=outputs)
model.summary()


# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")



@tf.function
def train_step(inputs, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.device("/gpu:1"):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = geom_cross_entropy(predictions, labels)
            loss = tf.reduce_sum(loss)/inputs[0].shape[0]    # use images.shape[0] instead of args.batch_size
        
        # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
    return loss

@tf.function
def test_step(inputs, labels):
    with tf.device("/gpu:1"):
        predictions = model(inputs, training=False)
        test_loss = geom_cross_entropy(predictions, labels)
        test_loss = tf.reduce_sum(test_loss)/inputs[0].shape[0]  # use images.shape[0] instead of args.batch_size
        test_accuracy.update_state(labels, predictions)
    return test_loss


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/hogPlusCNN/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)


step=0
for epoch in range(args.epochs):
    xtrain_epoch = xtrain.copy()
    random.shuffle(xtrain_epoch)
    
    print("\n\nEpoch {}: training on {} images and validating on {} images.\n".format(epoch+1, len(xtrain_epoch), len(xval)))
    while len(xtrain_epoch) > 0:
        tic = time.time()
        x_train = []
        y_train = []
        hog_train = []
        if len(xtrain_epoch) >= args.batch_size:
            train_viewpoints_batch = random.sample(xtrain_epoch, args.batch_size)
        else:
            train_viewpoints_batch = xtrain_epoch.copy()
        for example in train_viewpoints_batch:
            xtrain_epoch.remove(example)
        
        with Pool() as pool:
            train_batch = pool.map(render_train_view, train_viewpoints_batch)

        train_batch = np.array(train_batch)
        for i in range(len(train_batch)):
            x_train.append(train_batch[i, 0])
            y_train.append(train_batch[i, 1])
            hog_train.append(train_batch[i, 2])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        hog_train = np.array(hog_train)
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train, hog_train)).shuffle(100).batch(args.batch_size)
        
        # Save logs with TensorBoard Summary
        if step == 0:
            train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/hogPlusCNN/logs/train"
            val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d/hogPlusCNN/logs/val"
            train_summary_writer = tf.summary.create_file_writer(train_logdir)
            val_summary_writer = tf.summary.create_file_writer(val_logdir)
        
        for images, labels, hogFeatures in train_data:
            train_loss = train_step([images, hogFeatures], labels)
            step += 1
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss, step=step)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                #tf.summary.image("image", images, step=step, max_outputs=2)
            toc = time.time()
            print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                    train_loss, train_accuracy.result(), toc-tic))
            # Reset metrics for the next iteration
            train_accuracy.reset_states()
        

    epoch +=1
    
    test_it = 0
    test_loss = 0.
    for test_images, test_labels, test_features in tqdm(val_data, desc="Validation"):
        test_loss += test_step([test_images, test_features], test_labels)
        test_it +=1
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
        #tf.summary.image("val_images", test_images, step=epoch, max_outputs=2)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
    print(template.format(epoch, test_loss, test_accuracy.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_accuracy.reset_states()
