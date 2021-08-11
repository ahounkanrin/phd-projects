import tensorflow as tf
import numpy as np  
from tqdm import tqdm
import cv2 as cv
import argparse
import os
import time


INPUT_SIZE = (256, 256)
nclasses = 360
data_dir = "/scratch/hnkmah001/Datasets/ctfullbody/ctfullbody2d/train-val/"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    return parser.parse_args()

args = get_arguments()
     
# img_input = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1))
# x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(img_input)
# x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# x= tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same")(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

# x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same")(x)
# x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
# x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(x)
# x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
# x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(x)
# x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
# outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(x)

# model = tf.keras.Model(inputs=img_input, outputs=outputs)
# model.build(input_shape=(None, 256, 256, 1))
# model.summary()

with tf.name_scope("Encoder"):
    img_input = tf.keras.layers.Input(shape=(256, 256, 1))
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(img_input)
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(maxpool1)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(maxpool2)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(maxpool3)
    encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

with tf.name_scope("Decoder"):
    deconv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same")(encoded)
    upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv1)
    deconv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(upsample1)
    upsample2 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv2)
    deconv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(upsample2)
    upsample3 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv3)
    deconv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(upsample3)
    upsample4 = tf.keras.layers.UpSampling2D(size=(2,2))(deconv4)
    decoded = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(upsample4)


encoder = tf.keras.models.Model(img_input, encoded)
autoencoder = tf.keras.models.Model(img_input, decoded)

autoencoder.build(input_shape=(None, 256, 256, 1))
autoencoder.summary()


def crop_image(img):
    translation_window = [i for i in range(-20, 21, 5)]
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

@tf.function
def train_step(images):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation
    with tf.GradientTape() as tape:
        predictions = autoencoder(images, training=True)
        y_true = tf.reshape(images, [-1])
        y_pred = tf.reshape(predictions, [-1])
        loss = loss_object(y_true, y_pred)
        
    # Calculate gradients of cost function w.r.t. trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    train_loss.update_state(y_true, y_pred)
    #return loss

@tf.function
def test_step(images):
    predictions = autoencoder(images, training=False)
    y_true = tf.reshape(images, [-1])
    y_pred = tf.reshape(predictions, [-1])
    loss = loss_object(y_true, y_pred)
    test_loss.update_state(y_true, y_pred)
    #return test_loss

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(512, 512),
class_names=[str(i) for i in range(nclasses)],
label_mode="categorical",
shuffle=True,
color_mode = "grayscale",
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
color_mode="grayscale",
batch_size=args.batch_size)
val_data = val_data.map(lambda x,y: preprocess(x,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)


optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, decay=1e-6, momentum=0.9)

loss_object = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.MeanSquaredError(name="loss")
test_loss = tf.keras.metrics.MeanSquaredError(name="val_loss")


# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=autoencoder, optimizer=optimizer)
checkpoint_dir = "/scratch/hnkmah001/phd-projects/uncertainty-measure/one-class-classifier/checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=50)


# Save logs with TensorBoard Summary
train_logdir = "/scratch/hnkmah001/phd-projects/uncertainty-measure/one-class-classifier/logs/train"
val_logdir = "/scratch/hnkmah001/phd-projects/uncertainty-measure/one-class-classifier/logs/val"
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

# Training loop
step = 0
for epoch in range(args.epochs):
    for images, _ in train_data:
        tic = time.time()
        train_step(images)
        
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("accuracy", train_loss.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=1) 
        toc = time.time()
        print("Step {}: \t loss = {:.4f} \t ({:.2f} seconds/step)".format(step, 
                train_loss.result(), toc-tic))
        train_loss.reset_states()            

    test_it = 0
    for test_images, _ in tqdm(val_data, desc="Validation"):
        test_step(test_images)
    
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=1)

    ckpt_path = manager.save()
    template = "Epoch {}, Validation Loss: {:.4f}, ckpt {}\n\n"
    print(template.format(epoch+1, test_loss.result(), ckpt_path))
    
    # Reset metrics for the next epoch
    test_loss.reset_states()

