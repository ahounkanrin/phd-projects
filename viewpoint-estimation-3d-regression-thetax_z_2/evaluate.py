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
from utils import one_hot_encoding, euler_to_quaternion, quaternionAngle
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
    #x = theta * np.pi / 180.
    x = theta
    r11, r12, r13 = 1., 0. , 0.
    r21, r22, r23 = 0., np.cos(x), -np.sin(x)
    r31, r32, r33 = 0., np.sin(x), np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Ry(theta):
    #x = theta * np.pi / 180.
    x = theta
    r11, r12, r13 = np.cos(x), 0., np.sin(x)
    r21, r22, r23 = 0., 1., 0.
    r31, r32, r33 = -np.sin(x), 0, np.cos(x)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def Rz(theta):
    #x = theta * np.pi / 180.
    x = theta
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

#trainScan = trainScans[0]
#valScan = trainScans[1]
testScan = testScans[0]

# Load ct volume
INPUT_SIZE = (200, 200)
#imgpath_train = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(trainScan, trainScan)
#imgpath_val = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(valScan, valScan)
imgpath_test = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody/{}/{}.nii".format(testScan, testScan)

N = 512 
print("INFO: loading CT volume...")

ctVolume_test = nib.load(imgpath_test).get_fdata().astype(int)
ctVolume_test = np.squeeze(ctVolume_test)
voi_test = ctVolume_test[:,:, :N] # Extracts volume of interest from the full body ct volume
#voi = normalize(voi)    # Rescale CT numbers between 0 and 255
voi_test = voi_test - np.min(voi_test)
voi_test = np.pad(voi_test, N//2, "constant", constant_values=0)
voiShifted_test = np.fft.fftshift(voi_test)
del voi_test
voiFFT_test = np.fft.fftn(voiShifted_test)
del voiShifted_test
voiFFTShifted_test = np.fft.fftshift(voiFFT_test)
del voiFFT_test

print("3D FFT computed.")


# Rotation and Interpolation of the projection slice from the 3D FFT volume
x_axis = np.linspace(-N+0.5, N-0.5, 2*N)
y_axis = np.linspace(-N+0.5, N-0.5, 2*N)
z_axis = np.linspace(-N+0.5, N-0.5, 2*N)

projectionPlane = np.array([[xi, 0, zi] for xi in x_axis for zi in z_axis])
projectionPlane = np.reshape(projectionPlane, (2*N, 2*N, 3, 1), order="F")


def generate_test_data(viewpoint):
    theta_x = viewpoint[0]
    theta_y = viewpoint[1]
    theta_z = viewpoint[2]
    tx = 0
    ty = 0
    rotationMatrix = Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)
    projectionSlice = np.squeeze(rotate_plane(projectionPlane, rotationMatrix))
    projectionSliceFFT = interpn(points=(x_axis, y_axis, z_axis), values=voiFFTShifted_test, xi=projectionSlice, method="linear",
                                    bounds_error=False, fill_value=0)      
    img = np.abs(fftshift(ifft2(projectionSliceFFT)))
    img = img[N//2:N+N//2, N//2:N+N//2]
    img = normalize(img)
    img = img[56+tx:456+tx, 56+ty:456+ty]
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    label = np.array([theta_x, theta_y, theta_z])
    return img, label

def select_viewpoints(batch_size):
    batch = []
    for i in range(batch_size):
        theta_x = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        theta_y = np.random.uniform(low=-np.pi, high=np.pi)
        theta_z = np.random.uniform(low=-np.pi, high=np.pi)
        batch.append([theta_x, theta_y, theta_z])
    return batch


# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(4, activation=None)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs) #outputy,
model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
#train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

def test_step(images, labels):
    #with tf.device("/gpu:1"):
    predictions = model(images, training=False)
    predNorm = tf.broadcast_to(tf.norm(predictions, axis=-1, keepdims=True), shape=predictions.shape)
    predictions = tf.divide(predictions, predNorm)
    return predictions

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d-regression-quaternions/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=20)
checkpoint.restore(manager.checkpoints[-1])

step = 0
epoch = 0

tic = time.time()

x_test = []
y_test = []
test_viewpoints_batch = select_viewpoints(10)
with Pool() as pool:
    test_batch = pool.map(generate_test_data, test_viewpoints_batch)

test_batch = np.array(test_batch)
for i in range(len(test_batch)):
    x_test.append(test_batch[i, 0])
    y_test.append(test_batch[i, 1])

x_test = np.array(x_test)
y_test = np.array(y_test)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
preds_list = []
gt = []
for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Testing"):                                  
    preds = test_step(test_images, test_labels)
    preds_list.append(tf.squeeze(preds).numpy())
    #print(tf.squeeze(preds))
    test_labels = euler_to_quaternion(tf.squeeze(test_labels))
    #print(tf.squeeze(test_labels))
    gt.append(tf.squeeze(test_labels).numpy())

print(preds_list)
print(gt)
errors = [tf.squeeze(quaternionAngle(gt[i], preds_list[i])) for i in range(len(gt))]
#print(errors_z)
thresholds = np.array([theta for theta in range(0, 95, 10)])

print("\n\nMedian Error = {:.4f}".format(np.median(np.array(errors))))
with open("mederr.txt", "w") as f:
    print("Median Error = {:.4f}".format(np.median(np.array(errors))), file=f)

acc_theta = []

for theta in thresholds:
    acc_bool = np.array([errors[i] <= theta  for i in range(len(errors))])
    acc = np.mean(acc_bool)
    acc_theta.append(acc)
    #print("Accuracy at thetax = {} is: {:.4f}".format(theta, acc))

plt.figure(figsize=[8, 5])
#plt.title("Accuracy of the CNN")
#plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Threshold (bins of 10 degrees)")
plt.xticks(ticks=[i/10 for i in range(0, 95, 10)])
plt.yticks(ticks=[i/20 for i in range(21)])
plt.plot(thresholds//10, acc_theta, label=r"Rotation angle $\theta_x$")
#plt.plot(thresholds, acc_thetay, label=r"Rotation angle $\theta_y$")

plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("accuracy_reg_az_el_ip.png")