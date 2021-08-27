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

random.seed(0)
tf.random.set_seed(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training iteration")
    return parser.parse_args()
args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

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

trainScans = ["SMIR.Body.021Y.M.CT.57761", "SMIR.Body.025Y.M.CT.59477", "SMIR.Body.030Y.F.CT.59466", 
             "SMIR.Body.030Y.F.CT.59471", "SMIR.Body.033Y.M.CT.57764", "SMIR.Body.036Y.F.CT.58319", 
             "SMIR.Body.037Y.M.CT.57613", "SMIR.Body.037Y.M.CT.59473", "SMIR.Body.041Y.F.CT.57699", 
             "SMIR.Body.043Y.M.CT.58317", "SMIR.Body.045Y.M.CT.59467", "SMIR.Body.045Y.M.CT.59476", 
             "SMIR.Body.045Y.M.CT.59481", "SMIR.Body.047Y.F.CT.57792", "SMIR.Body.049Y.M.CT.59482", 
             "SMIR.Body.052Y.M.CT.57765", "SMIR.Body.052Y.M.CT.59475",  "SMIR.Body.057Y.F.CT.57793", 
             "SMIR.Body.057Y.M.CT.57609", "SMIR.Body.057Y.M.CT.59483", "SMIR.Body.058Y.M.CT.57767"]
    
testScans = ["SMIR.Body.025Y.M.CT.57697", "SMIR.Body.033Y.M.CT.57766", "SMIR.Body.037Y.F.CT.57796", 
            "SMIR.Body.040Y.M.CT.57768", "SMIR.Body.045Y.M.CT.59470", "SMIR.Body.049Y.M.CT.57791", 
            "SMIR.Body.056Y.F.CT.59474", "SMIR.Body.057Y.F.CT.59693"]

trainScan = trainScans[0]
valScan = trainScans[1]
testScan = testScans[0]

# Load ct volume
INPUT_SIZE = (200, 200)
imgpath_train = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(trainScan, trainScan)
imgpath_val = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(valScan, valScan)
imgpath_test = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(testScan, testScan)

N = 512 
print("INFO: loading CT volume...")
ctVolume = nib.load(imgpath_train).get_fdata().astype(int)
ctVolume = np.squeeze(ctVolume)
voi = ctVolume[:,:, :N] # Extracts volume of interest from the full body ct volume
#voi = normalize(voi)    # Rescale CT numbers between 0 and 255
voi = voi - np.min(voi)
voi = np.pad(voi, N//2, "constant", constant_values=0)
voiShifted = np.fft.fftshift(voi)
del voi
voiFFT = np.fft.fftn(voiShifted)
del voiShifted
voiFFTShifted = np.fft.fftshift(voiFFT)
del voiFFT

ctVolume_val = nib.load(imgpath_val).get_fdata().astype(int)
ctVolume_val = np.squeeze(ctVolume)
voi_val = ctVolume_val[:,:, :N] # Extracts volume of interest from the full body ct volume
#voi = normalize(voi)    # Rescale CT numbers between 0 and 255
voi_val = voi_val - np.min(voi_val)
voi_val = np.pad(voi_val, N//2, "constant", constant_values=0)
voiShifted_val = np.fft.fftshift(voi_val)
del voi_val
voiFFT_val = np.fft.fftn(voiShifted_val)
del voiShifted_val
voiFFTShifted_val = np.fft.fftshift(voiFFT_val)
del voiFFT_val

print("3D FFT computed.")


# Rotation and Interpolation of the projection slice from the 3D FFT volume
x_axis = np.linspace(-N+0.5, N-0.5, 2*N)
y_axis = np.linspace(-N+0.5, N-0.5, 2*N)
z_axis = np.linspace(-N+0.5, N-0.5, 2*N)

projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")

def generate_train_data(viewpoint):
    theta_x = viewpoint[0]
    theta_y = viewpoint[1]
    theta_z = viewpoint[2]
    #tx = viewpoint[1]
    #ty = viewpoint[2]
    rotationMatrix = Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)
    projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=voiFFTShifted, xi=projectionSlice, method="linear",
                                    bounds_error=False, fill_value=0)      
    img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    img = img[N//2:N+N//2, N//2:N+N//2]
    img = normalize(img)
    #img = img[56+tx:456+tx, 56+ty:456+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = np.array([theta_x, theta_y, theta_z])
    return img, label

def generate_val_data(viewpoint):
    theta_x = viewpoint[0]
    theta_y = viewpoint[1]
    theta_z = viewpoint[2]
    #tx = viewpoint[1]
    #ty = viewpoint[2]
    rotationMatrix = Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)
    projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=voiFFTShifted_val, xi=projectionSlice, method="linear",
                                    bounds_error=False, fill_value=0)      
    img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    img = img[N//2:N+N//2, N//2:N+N//2]
    img = normalize(img)
    #img = img[56+tx:456+tx, 56+ty:456+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = np.array([theta_x, theta_y, theta_z])
    return img, label

def select_viewpoints(batch_size):
    batch = []
    for i in range(batch_size):
        theta_x = np.random.uniform(low=-90, high=90) 
        theta_y = 0 # np.random.uniform(low=-30, high=30)
        theta_z = np.random.uniform(low=-180, high=180)
        batch.append([theta_x, theta_y, theta_z])
    return batch


# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputx = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)(x)
outputx = tf.multiply(outputx, tf.constant(90.0))
#outputy = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)(x)
#outputy = tf.multiply(outputy, tf.constant(30.0))
outputz = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)(x)
outputz = tf.multiply(outputz, tf.constant(180.0))
model = tf.keras.Model(inputs=inputs, outputs=[outputx,  outputz]) #outputy,
model.summary()

# Define cost function, optimizer and metrics
loss_objectx = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
loss_objecty = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
loss_objectz = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)

test_lossx = tf.keras.metrics.MeanSquaredError()
#test_lossy = tf.keras.metrics.MeanSquaredError()
test_lossz = tf.keras.metrics.MeanSquaredError()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
#train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    #with tf.device("/gpu:1"):
    with tf.GradientTape() as tape:
        predx, predz = model(images, training=True)
        lossx = loss_objectx(labels, predx)
        #lossy = loss_objecty(labels, predy)
        lossz = loss_objectz(labels, predz)
        loss = tf.divide(lossx, tf.constant(180.0**2)) + tf.divide(lossz, tf.constant(360.0**2)) # + tf.divide(lossy, tf.constant(60.0**2)) 
        #loss = tf.reduce_sum(loss)/images.shape[0]    # use images.shape[0] instead of args.batch_size
    
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(images, labels):
    #with tf.device("/gpu:1"):
    predx, predz = model(images, training=True)
    test_lossx.update_state(labels, predx)
    #test_lossy.update_state(labels, predy)
    test_lossz.update_state(labels, predz)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-thetax_thetay/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)

step = 0
epoch = 0
while True:
    tic = time.time()
    x_train = []
    y_train = []
    train_viewpoints_batch = select_viewpoints(args.batch_size)
    with Pool() as pool:
        train_batch = pool.map(generate_train_data, train_viewpoints_batch)

    train_batch = np.array(train_batch)
    for i in range(len(train_batch)):
        x_train.append(train_batch[i, 0])
        y_train.append(train_batch[i, 1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    
    # Save logs with TensorBoard Summary
    if step == 0:
        train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-thetax_thetay/logs/train"
        val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-thetax_thetay/logs/val"
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        val_summary_writer = tf.summary.create_file_writer(val_logdir)
    
    for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        train_loss = train_step(images, labels)
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss, step=step)
            #tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=2)
        toc = time.time()
        print("Step {}: \t loss = {:.4f}  \t ({:.2f} seconds/step)".format(step, 
                train_loss, toc-tic))
        # Reset metrics for the next iteration
        #train_accuracy.reset_states()
    
    if step % 5000 == 0:
        epoch += 1

        x_val = []
        y_val = []
        val_viewpoints_batch = select_viewpoints(200*args.batch_size)
        with Pool() as pool:
            val_batch = pool.map(generate_val_data, val_viewpoints_batch)

        val_batch = np.array(val_batch)
        for i in range(len(val_batch)):
            x_val.append(val_batch[i, 0])
            y_val.append(val_batch[i, 1])

        x_val = np.array(x_val)
        y_val = np.array(y_val)
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

        for test_images, test_labels in tqdm(val_data.map(preprocess, 
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            test_step(test_images, test_labels)
        test_loss = test_lossx.result()/(90.0**2)  + test_lossz.result()/(360.0**2)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", test_loss, step=epoch)
            #tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
            tf.summary.image("val_images", test_images, step=epoch, max_outputs=2)

        ckpt_path = manager.save()
        template = "Epoch {}, Validation Loss: {:.4f}, ckpt {}\n\n"
        print(template.format(epoch, test_loss, ckpt_path))
        
        # Reset metrics for the next epoch
        #test_accuracy.reset_states()
        test_lossx.reset_states()
        #test_lossy.reset_states()
        test_lossz.reset_states()
