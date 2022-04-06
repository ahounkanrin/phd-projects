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
from utils import euler_to_quaternion, quaternionLoss

# tf.config.threading.set_inter_op_parallelism_threads(8)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.set_soft_device_placement(False)

print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)

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
    tx = np.random.choice(translation_window)
    ty = np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(imgpath, label):
    img = tf.map_fn(load_img, imgpath, fn_output_signature=tf.uint8)
    img = tf.map_fn(crop_image, img)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, tf.constant(255.0, dtype=tf.float32))
    return img, label

@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        rawpreds = model(images, training=True)
        loss = quaternionLoss(rawpreds, labels)
        loss = tf.reduce_sum(loss)/images.shape[0]    # use images.shape[0] instead of args.batch_size
    
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(images, labels):
    rawpreds = model(images, training=True)
    loss = quaternionLoss(rawpreds, labels)
    loss = tf.reduce_sum(loss)/images.shape[0]
    return loss

# Load dataset
data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/roll_pitch_yaw/train-val/"
train_df = pd.read_csv("train.csv", sep=",")
img_path = train_df["image"].apply(lambda imgID: data_dir+imgID)
label_roll = (np.pi/180) * train_df["azimuth"].astype(int)
label_pitch = (np.pi/180) * train_df["elevation"].astype(int)
label_yaw = (np.pi/180) * train_df["inplane"].astype(int)
img_path_list = tf.constant( np.array(img_path))
labels_list = tf.constant(np.array([euler_to_quaternion([label_roll[i], label_pitch[i], label_yaw[i]]) for i in range(len(label_yaw))]))
dataset = tf.data.Dataset.from_tensor_slices((img_path_list, labels_list)).shuffle(len(img_path_list))
val_data = dataset.take(int(0.2 * len(img_path_list))).batch(args.batch_size)
train_data = dataset.skip(int(0.2 * len(img_path_list))).batch(args.batch_size)


# Define model
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
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Define checkpoint manager to save model weights
#checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
#checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d-regression3/checkpoints/"
#if not os.path.isdir(checkpoint_dir):
#    os.mkdir(checkpoint_dir)
#manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)


# Save logs with TensorBoard Summary
#train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d-regression3/logs/train"
#val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation-3d-regression3/logs/val"
#train_summary_writer = tf.summary.create_file_writer(train_logdir)
#val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        tic = time.time()
        train_loss = train_step(images, labels)
        #print(gradients)
        step += 1
        #with train_summary_writer.as_default():
        #    tf.summary.scalar("loss", train_loss, step=step)
        #    tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        print("Step {}: \t loss = {:.6f}  \t({:.2f} seconds/step)".format(step, train_loss, toc-tic))

    test_it = 0
    test_loss = 0.
    for test_images, test_labels in tqdm(val_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE), 
                                                        desc="Validation"):
        test_loss += test_step(test_images, test_labels)
        test_it +=1
        
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    #with val_summary_writer.as_default():
    #    tf.summary.scalar("val_loss", test_loss, step=epoch)
    #    tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    #ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.6f},\n\n"
    print(template.format(epoch+1, test_loss))
    
