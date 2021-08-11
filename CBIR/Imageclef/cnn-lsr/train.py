import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import os
import time
import pandas as pd 
from utils import geom_cross_entropy
import h5py
import cv2 as cv

print("INFO: Processing dataset...")
INPUT_SIZE = (256, 256)

train_df = pd.read_csv("train.csv", sep=";")
test_df = pd.read_csv("test.csv", sep=";")
classes = sorted(list(set(train_df["irma_code"])))
nclasses = len(classes)
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument("--cnn", choices=["densenet", "inception", "resenet", "vgg"], required=True, help="Model architecture")
    return parser.parse_args()

args = get_arguments()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    return x, tf.one_hot(y, nclasses)

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

# Load dataset
def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')
    x_test = hf.get('x_test')
    y_test = hf.get('y_test')

    x_train = np.array(x_train)
    y_train = np.array(y_train).astype(int)
    x_test = np.array(x_test)
    y_test = np.array(y_test).astype(int)
    return x_train, y_train, x_test, y_test

x1, y1, x2, y2 = read_dataset(DATA_DIR+"imageclef.h5")

x_train = [0] * len(x1)
x_test = [0] * len(x2)
y_train = y1
y_test = y2

clahe = cv.createCLAHE(clipLimit=2.55, tileGridSize=(8,8))
for i in tqdm(range(len(x1))):
    img = np.squeeze(x1[i])
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    # img_r = img
    # img_g = clahe.apply(img)
    # img_b = cv.fastNlMeansDenoising(img, h=2, templateWindowSize=4, searchWindowSize=4)
    x_train[i] = np.stack([img, img, img], axis=-1)
x_train = np.asarray(x_train)
print("shape of x_train:", x_train.shape)

for i in tqdm(range(len(x2))):
    img = np.squeeze(x2[i])
    img = cv.resize(img, INPUT_SIZE, interpolation=cv.INTER_AREA)
    # img_r = img
    # img_g = clahe.apply(img)
    # img_b = cv.fastNlMeansDenoising(img, h=2, templateWindowSize=4, searchWindowSize=4)
    x_test[i] = np.stack([img, img, img], axis=-1)
x_test = np.asarray(x_test)
print("shape of x_test", x_test.shape)

# x_train = tf.constant(x_train / 255.0, dtype=tf.float32)
# x_test = tf.constant(x_test / 255.0, dtype=tf.float32)

all_train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
train_data = all_train_data.take(int(0.8*len(x_train))).batch(args.batch_size)
val_data = all_train_data.skip(int(0.8*len(x_train))).batch(args.batch_size)
#test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

if args.cnn == "densenet":
    baseModel = tf.keras.applications.DenseNet121(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "inception":
    baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "resenet":
    baseModel = tf.keras.applications.ResNet101(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")
elif args.cnn == "vgg":
    baseModel = tf.keras.applications.VGG16(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), include_top=False, weights="imagenet")

inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
x = baseModel(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
outputs = tf.keras.layers.Dense(nclasses, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/CBIR/Imageclef/cnn-lsr/{}/checkpoints/".format(args.cnn)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=50)


# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/CBIR/Imageclef/cnn-lsr/{}/logs/train".format(args.cnn)
val_logdir = "/scratch/hnkmah001/phd-projects/CBIR/Imageclef/cnn-lsr/{}/logs/val".format(args.cnn)
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    for images, labels in train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        tic = time.time()
        train_loss = train_step(images, labels)
        
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss, step=step)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        print("Step {}: \t loss = {:.4f} \t acc = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                train_loss, train_accuracy.result(), toc-tic))
        train_accuracy.reset_states()     

    test_it = 0
    test_loss = 0.
    for test_images, test_labels in tqdm(val_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE), 
                                        desc="Validation"):
        test_loss += test_step(test_images, test_labels)
        test_it +=1

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
