import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
from matplotlib import pyplot as plt
import cv2 as cv
import os
import time



print("INFO: Processing dataset...")
INPUT_SIZE = (200, 200)
nclasses = 360
data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/train-val/"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()

translation_window = [i for i in range(-10, 11, 2)]

def crop_image(img):
    tx = np.random.choice(translation_window)
    ty = np.random.choice(translation_window)
    img = tf.image.crop_to_bounding_box(img, offset_height=56+ty, offset_width=56+tx, target_height=400, target_width=400)
    img = tf.image.resize(img, size=INPUT_SIZE, method="nearest")
    return img

def preprocess(x, y):
    x = tf.map_fn(crop_image, x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, y

def myMSE(predictions, labels):
    loss_batch = tf.math.square(labels - predictions)
    loss = tf.math.reduce_mean(loss_batch)
    return loss

alpa = 0
def myMSE2(predictions, labels):
    prod = tf.math.reduce_sum(tf.math.multiply(labels, predictions))
    loss = tf.math.square(1 - prod)
    return loss    

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        predictions = tf.squeeze(predictions)
        loss = myMSE2(predictions, labels)
           
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
    
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    predictions = tf.squeeze(predictions)
    return myMSE2(predictions, labels)


#with strategy.scope():

# Load dataset


train_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(512, 512),
class_names=[str(i) for i in range(nclasses)],
label_mode="categorical",
shuffle=True,
batch_size=args.batch_size)
train_data = train_data.map(lambda x,y: (preprocess(x,y)), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
train_data = train_data.prefetch(1024)
#train_data = strategy.experimental_distribute_dataset(train_data)

val_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(512, 512),
class_names=[str(i) for i in range(nclasses)],
label_mode="categorical",
shuffle=True,
batch_size=args.batch_size)
val_data = val_data.map(lambda x,y: preprocess(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
#val_data = strategy.experimental_distribute_dataset(val_data)

baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Define cost function, optimizer and metrics
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/regression/exp1-no-aug/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=50)


# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/regression/exp1-no-aug/logs/train"
val_logdir = "/scratch/hnkmah001/phd-projects/viewpoint-estimation2/regression/exp1-no-aug/logs/val"
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    for images, labels in train_data:
        tic = time.time()
        train_loss = train_step(images, labels)
        
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss, step=step)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        print("Step {}: \t loss = {:.8f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                train_loss, train_accuracy.result(), toc-tic))
        train_accuracy.reset_states()            

    test_it = 0
    test_loss = 0.
    for test_images, test_labels in tqdm(val_data, desc="Validation"):
        test_loss += test_step(test_images, test_labels)
        test_it +=1
    test_loss = test_loss/tf.constant(test_it, dtype=tf.float32)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss, step=epoch)
        tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.8f}, Validation Accuracy: {:.4f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss, test_accuracy.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_accuracy.reset_states()

