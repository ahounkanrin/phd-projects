import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
from utils import geom_cross_entropy, geom_cross_entropy_el
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
    tx = np.random.choice(translation_window)
    ty = np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(x, y1, y2):
    x = tf.map_fn(crop_image, x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, tf.one_hot(y1, nclasses//2), tf.one_hot(y2, nclasses)

@tf.function
def train_step(images, labelsx, labelsz):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        predx, predz = model(images, training=True)
        assert predx.shape == labelsx.shape
        lossx, lossz = geom_cross_entropy_el(predx, labelsx), geom_cross_entropy(predz, labelsz)
        loss = tf.reduce_sum(lossx)/images.shape[0] + 0.5*tf.reduce_sum(lossz)/images.shape[0]
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy_elevation.update_state(labelsx, predx)
    train_accuracy_azimuth.update_state(labelsz, predz)
    return loss, gradients

@tf.function
def test_step(images, labelsx, labelsz):
    predx, predz = model(images, training=False)
    lossx, lossz = geom_cross_entropy_el(predx, labelsx), geom_cross_entropy(predz, labelsz)
    test_loss = tf.reduce_sum(lossx)/images.shape[0] + 0.5*tf.reduce_sum(lossz)/images.shape[0]
    test_accuracy_elevation.update_state(labelsx, predx)
    test_accuracy_azimuth.update_state(labelsz, predz)
    return test_loss

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

assert max(y_train_az) == 35 and max(y_train_el) == 17

for i in tqdm(range(len(x1))):
    img = np.squeeze(x1[i])
    x_train[i] = np.stack([img, img, img], axis=-1)
x_train = np.asarray(x_train)
print("shape of x_train:", x_train.shape)

for i in tqdm(range(len(x2))):
    img = np.squeeze(x2[i])
    x_test[i] = np.stack([img, img, img], axis=-1)
x_test = np.asarray(x_test)
print("shape of x_test", x_test.shape)


all_train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_el, y_train_az)).shuffle(len(x_train))
train_data = all_train_data.take(int(0.8*len(x_train))).batch(args.batch_size)
val_data = all_train_data.skip(int(0.8*len(x_train))).batch(args.batch_size)
train_data = train_data.map(lambda x,y1, y2: (preprocess(x,y1, y2)), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
val_data = val_data.map(lambda x,y1, y2: preprocess(x, y1, y2), num_parallel_calls=tf.data.experimental.AUTOTUNE)

#print(train_data)

# Define model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs_el = tf.keras.layers.Dense(nclasses//2, activation="softmax")(x)
outputs_az = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=[outputs_el, outputs_az])

model.summary()

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


# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-classification-thetax_z/logs/train"
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-classification-thetax_z/logs/val"
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    for images, labelsx, labelsz in train_data:
        tic = time.time()
        train_loss, gradients = train_step(images, labelsx, labelsz)
        #print(gradients)
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss, step=step)
            tf.summary.scalar("accuracy_elevation", train_accuracy_elevation.result(), step=step)
            tf.summary.scalar("accuracy_azimuth", train_accuracy_azimuth.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=1) 
            tf.summary.histogram("gradients_input_layer", gradients[0], step=step)
            tf.summary.histogram("gradients_output_layer", gradients[-1], step=step)
        toc = time.time()
        print("Step {}: \t loss = {:.4f} \t accx = {:.4f} \t  accz = {:.4f} \t({:.2f} seconds/step)".format(step, 
                train_loss, train_accuracy_elevation.result(), train_accuracy_azimuth.result(), toc-tic))
        train_accuracy_elevation.reset_states()    
        train_accuracy_azimuth.reset_states()        

    test_it = 0
    test_loss = 0.
    for test_images, test_labelsx, test_labelsz in tqdm(val_data, desc="Validation"):
        test_loss += test_step(test_images, test_labelsx, test_labelsz)
        test_it +=1
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.scalar("val_accuracy_elevation", test_accuracy_elevation.result(), step=epoch)
        tf.summary.scalar("val_accuracy_azimuth", test_accuracy_azimuth.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.4f},\t accx = {:.4f} \t  accz = {:.4f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss, test_accuracy_elevation.result(), test_accuracy_azimuth.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_accuracy_elevation.reset_states()
    test_accuracy_azimuth.reset_states()

