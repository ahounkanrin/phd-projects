import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from matplotlib import pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=1,   type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.000001, type=float, help="Initial learning rate")
    parser.add_argument("--is_training", default=True, type=lambda x: bool(int(x)), help="Training or testing mode")
    return parser.parse_args()

def read_dataset(hf5):
    hf = h5py.File(hf5,'r')
    x_train = np.array(hf.get('x_train'))
    y_train = np.array(hf.get('y_train'))
    x_val = np.array(hf.get("x_val"))
    y_val = np.array(hf.get("y_val"))
    x_test = np.array(hf.get('x_test'))
    y_test = np.array(hf.get('y_test'))
    return x_train, y_train, x_val, y_val, x_test, y_test

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y= tf.cast(y, dtype=tf.float32)
    x = tf.divide(x, tf.constant(255.0, dtype=tf.float32))
    y = tf.constant(np.pi/180.0, dtype=tf.float32) * y
    return x, y

def myMSE(images, labels):
    gt_sin = tf.math.sin(labels)
    gt_cos = tf.math.cos(labels)
    predictions = model(images)
    pred_sin = tf.math.sin(predictions)
    pred_cos = tf.math.cos(predictions)
    loss_sin = tf.math.squared_difference(gt_sin, pred_sin)
    loss_sin = tf.math.reduce_mean(loss_sin)
    loss_cos = tf.math.squared_difference(gt_cos, pred_cos)
    loss_cos = tf.math.reduce_mean(loss_cos)
    loss = loss_cos + loss_sin
    return loss

def myMSE2(predictions, labels):
    loss_batch = 1.0 - tf.math.cos(labels - predictions)
    loss = tf.math.reduce_mean(loss_batch)
    return loss

args = get_arguments()


# Load dataset
DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"
x_train, y_train, x_val, y_val, x_test, y_test = read_dataset(DIR+'chest_fov_400x400_sparse_labels.h5')

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size) 
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

print("INFO: Datasets loaded...")

# Define the model

baseModel = tf.keras.applications.InceptionV3(input_shape=(400, 400, 3), include_top=False, weights="imagenet")
#baseModel.trainable = False
x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu)(x) 
x = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)
outputs = tf.multiply(x, tf.constant(2*np.pi))
model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
model.summary()

# Define cost function, optimizer and metrics
loss_object = tf.keras.losses.MeanSquaredError()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=1000, 
                                                            decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
train_loss = tf.keras.metrics.MeanSquaredError(name="train_loss")
test_loss = tf.keras.metrics.MeanSquaredError(name="test_loss")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = myMSE2(predictions, labels)
           
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #train_loss_sin.update_state(tf.math.sin(labels), tf.math.sin(predictions_sin))
    return loss
    
@tf.function
def test_step(images, labels):
    predictions = model(images)
    return myMSE2(predictions, labels)

@tf.function 
def train_step2(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions) 

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(labels, predictions)

@tf.function
def test_step2(images, labels):
    predictions = model(images)
    test_loss.update_state(labels, predictions)

    
# Define checkpoint manager to save model weights
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = "./checkpoints/"
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)

if args.is_training:

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
        for images, labels in tqdm(train_data.map(preprocess, 
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Training"):
            loss = train_step(images, labels)
            print("\t Training loss: {:.4f}".format(loss))
            if step == 0:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="InceptionV3", step=0)
            step += 1
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=step)
                tf.summary.image("image", images, step=step, max_outputs=8)

        test_it = 0
        loss_val = 0.
        for test_images, test_labels in tqdm(val_data.map(preprocess, 
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            loss_val += test_step(test_images, test_labels)
            test_it += 1
        loss_val = loss_val/tf.constant(test_it, dtype=tf.float32)
            
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", loss_val, step=epoch)
            tf.summary.image("val_images", test_images, step=epoch, max_outputs=8)

        ckpt_path = manager.save()
        template = "\n\n\nEpoch {}, Val Loss: {:.4f},  ckpt {}\n\n"
        print(template.format(epoch+1, loss_val, ckpt_path))
        
        # Reset metrics for the next epoch
       
        #test_loss.reset_states()        
else:

    checkpoint.restore(manager.checkpoints[-1])

    pred = []
    test_loss = 0.
    count = 0
    for test_images, test_labels in tqdm(test_data.map(preprocess, 
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE), desc="Validation"):
            test_loss += test_step(test_images, test_labels)
            pred.append((180./np.pi)*model(test_images)) # Convert angles from radians to degrees
            count += 1
    test_loss = test_loss/tf.constant(count, dtype=tf.float32)
    print("Test Loss: {:.4f}".format(test_loss))

    gt = y_test.astype(float)
    pred = np.array(pred)
    pred = np.squeeze(pred)
    pred_err1 = np.abs(pred - gt) 
    pred_err2 = np.abs(-360 + pred - gt)
    pred_err3 = np.abs(360 + pred - gt)
    thresholds = [theta for theta in range(0, 60, 5)]
    acc_list = []

    print("gt:", gt)
    print("pred:", pred)
    
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
    
    