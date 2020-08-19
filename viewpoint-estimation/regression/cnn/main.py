import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt


def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')
    x_val = hf.get("x_val")
    y_val = hf.get("y_val")
    x_test = hf.get('x_test')
    y_test = hf.get('y_test')

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

args = get_arguments()

# Load dataset
DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"
x_train, y_train, x_val, y_val, x_test, y_test = read_dataset(DIR+'chest_fov_400x400_sparse_labels.h5')

x_train = tf.constant(x_train/255.0, dtype=tf.float32)
x_val = tf.constant(x_val/255.0, dtype=tf.float32)
x_test = tf.constant(x_test/255.0, dtype=tf.float32)
y_train = tf.cast(y_train, dtype=tf.float32)
y_val = tf.cast(y_val, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size) 
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

print("[INFO] Datasets created...")

# Define the model

baseModel = tf.keras.applications.InceptionV3(input_shape=(400, 400, 3), include_top=False, weights="imagenet")
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)
outputs = tf.multiply(x, 360)

model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)


# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.MeanSquaredError()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.MeanSquaredError(name="train_loss")
test_loss = tf.keras.metrics.MeanSquaredError(name="test_loss")

@tf.function
def train_step(images, labels):
    # All ops involving trainable variables under the GradientTape context manager are recorded for gradient computation purposes
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Calculate metrics
    train_loss.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    test_loss.update_state(labels, predictions)

# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "./checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)

if args.is_training:

    #manager.restore_or_initialize() 
    # Save logs with TensorBoard Summary
    train_logdir = "./logs/train"
    val_logdir = "./logs/val"
    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)

    with train_summary_writer.as_default():
        tf.summary.histogram(name="train_classes", data=np.argmax(y_train), step=0)
    with val_summary_writer.as_default():
        tf.summary.histogram(name="val_classes", data=np.argmax(y_val), step=0)
    tf.summary.trace_on(graph=True)    

    # Training loop
    step = 0
    for epoch in tqdm(range(args.epochs)):
        for images, labels in tqdm(train_data, desc="Training"):
            train_step(images, labels)
            if step == 0:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="InceptionV3", step=0)
            step += 1
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=step)
                tf.summary.image("image", images, step=step, max_outputs=8)

        for test_images, test_labels in tqdm(val_data, desc="Validation"):
            test_step(test_images, test_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
            tf.summary.image("val_images", test_images, step=epoch, max_outputs=8)

        ckpt_path = manager.save()
        template = "\nEpoch {}, Loss: {:.4f}, Validation Loss: {:.4f}, ckpt {}\n\n"
        print(template.format(epoch+1, train_loss.result(), test_loss.result(), ckpt_path))
        
        # Reset metrics for the next epoch
        train_loss.reset_states()
        test_loss.reset_states()
        
else:

    checkpoint.restore(manager.checkpoints[-1])

    for val_images, val_labels in tqdm(val_data, desc="Validation"):
            test_step(val_images, val_labels)
    template = "Validation Loss: {:.4f}"
    print(template.format(test_loss.result()))
    test_loss.reset_states()

    pred = []
    for test_images, test_labels in tqdm(test_data, desc="Validation"):
            test_step(test_images, test_labels)
            pred.append(model(test_images))
    template = "Test Loss: {:.4f}"
    print(template.format(test_loss.result()))

    
    gt = y_test
    #print("GROUND TRUTH:", gt)
    #print("PREDICTIONS:", pred)


    pred_err1 = np.abs(np.array(pred) - np.array(gt)) 
    pred_err2 = np.abs(-360 + np.array(pred) - np.array(gt))
    pred_err3 = np.abs(360 + np.array(pred) - np.array(gt))
    thresholds = [theta for theta in range(0, 60, 5)]
    acc_list = []
    #theta = 10
    for theta in thresholds:

        acc_bool = np.array([pred_err1[i] <= theta or pred_err2[i] <= theta or pred_err3[i] <= theta for i in range(len(pred_err1))])

        acc = np.array([int(i) for i in acc_bool])
        acc = np.mean(acc)
        acc_list.append(acc)
        print("Accuracy at theta = {} is: {:.4f}".format(theta, acc))

        
    plt.figure()
    plt.scatter(thresholds, acc_list)
    plt.grid(True)
    #plt.show()
    plt.savefig("accuracy.png")
    
    