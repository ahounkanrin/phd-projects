import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from network import myModel


def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = np.array(hf.get('x_train'))
    y_train = np.array(hf.get('y_train')).astype(int)
    x_test = np.array(hf.get('x_test'))
    y_test = np.array(hf.get('y_test')).astype(int)
    return x_train, y_train, x_test, y_test

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate")
    return parser.parse_args()


args = get_arguments()

# Load dataset
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'
x_train, y_train, x_test, y_test = read_dataset(DATA_DIR+"imageclef.h5")
x_train = tf.constant(x_train/255.0, dtype=tf.float32)
x_test = tf.constant(x_test/255.0, dtype=tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size) 
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

# Define the model
model = myModel()
model.build((None, 512, 512, 1))
model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name="loss")
test_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name="test_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(images, labels):
    # All ops involving trainables under the GradientTape context manager are recorded for gradient computation purposes
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Calculate metrics
    train_accuracy.update_state(labels, predictions)
    train_loss.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    test_loss.update_state(labels, predictions)
    test_accuracy.update_state(labels, predictions)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "./checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)

# Save logs with TensorBoard Summary
train_logdir = "./logs/train"
val_logdir = "./logs/val"
train_summary_writer = tf.summary.create_file_writer(train_logdir)
val_summary_writer = tf.summary.create_file_writer(val_logdir)

with train_summary_writer.as_default():
    tf.summary.histogram(name="train_classes", data=y_train, step=0)
with val_summary_writer.as_default():
    tf.summary.histogram(name="test_classes", data=y_test, step=0)
tf.summary.trace_on(graph=True)    


# Training loop
step = 0
for epoch in tqdm(range(args.epochs)):
    for images, labels in tqdm(train_data, desc="training"):
        train_step(images, labels)
        if step == 0:
            with train_summary_writer.as_default():
                tf.summary.trace_export(name="vgg16", step=0)
        step += 1
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=step)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
            tf.summary.image("image", images, step=step, max_outputs=8)

    for test_images, test_labels in tqdm(test_data, desc="testing"):
        test_step(test_images, test_labels)
    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
        tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
        tf.summary.image("val_images", test_images, step=epoch, max_outputs=8)

    ckpt_path = manager.save()
    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, ckpt {}"
    print(template.format(epoch+1, train_loss.result(), tf.round(train_accuracy.result()*100),
                        test_loss.result(), tf.round(test_accuracy.result()*100), ckpt_path))
    
    # Reset metrics for the next epoch
    #train_accuracy.reset_states()
    #test_loss.reset_states()
    #test_accuracy.reset_states()

