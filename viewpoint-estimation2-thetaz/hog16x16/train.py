import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time
from utils import geom_cross_entropy
from skimage.feature import hog

#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

print("INFO: Processing dataset...")
INPUT_SIZE = (400, 400)
IMAGE_SIZE = (512, 512)
nclasses = 360
data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/train-val/"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

translation_window = [i for i in range(-20, 21, 2)]

def crop_image(img):
    tx = np.random.choice(translation_window)
    ty = np.random.choice(translation_window)
    img = img[56+tx:456+tx, 56+ty:456+ty]
    return img

def preprocess(x, y):
    x = tf.map_fn(crop_image, x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

def myHog(x):
    return hog(x, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, multichannel=None)

@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = geom_cross_entropy(predictions, labels)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=images.shape[0]) # strategy.num_replicas_in_sync*images.shape[0]
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    test_loss = geom_cross_entropy(predictions, labels)
    test_loss = tf.reduce_sum(test_loss, axis=-1)  
    test_loss = tf.nn.compute_average_loss(test_loss, global_batch_size=images.shape[0]) #strategy.num_replicas_in_sync*images.shape[0]
    test_accuracy.update_state(labels, predictions)
    return test_loss



# Load dataset

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.2,
    dtype=None,
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=args.batch_size,
    shuffle=True,
    seed=123,
    subset="training",
    interpolation="nearest",
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=args.batch_size,
    shuffle=False,
    seed=123,
    subset="validation",
    interpolation="nearest",
)


inputs = tf.keras.Input(shape=(20736,)) # |   | (900,) | (4356,)
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
outputs = tf.keras.layers.Dense(360, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
#model.build((None, 4356,))
model.summary()


# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/hog/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=50)


# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/hog/logs/train"
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/hog/logs/val"
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    batches = 0 
    for images, labels in train_data:
        tic = time.time()
        train_loss = train_step(tf.map_fn(myHog, tf.map_fn(crop_image, images)), labels)
        
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss, step=step)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                train_loss, train_accuracy.result(), toc-tic))
        train_accuracy.reset_states() 
        batches += 1
        if batches > 5760/args.batch_size:
            break   # Needs to break manually because the generator loops indefinitely


    test_it = 0
    test_loss = 0.
    for test_images, test_labels in tqdm(val_data, desc="Validation"):
        test_loss += test_step(tf.map_fn(myHog,tf.map_fn(crop_image, test_images)), test_labels)
        test_it +=1
        if test_it > 1440/args.batch_size:
            break
            
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss, test_accuracy.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_accuracy.reset_states()

