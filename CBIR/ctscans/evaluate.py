import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from network import myModel

strategy = tf.distribute.MirroredStrategy()

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
    return parser.parse_args()


args = get_arguments()

# Load dataset
DATA_DIR = '/scratch/hnkmah001/Datasets/ImageCLEF09/'
x_train, y_train, x_test, y_test = read_dataset(DATA_DIR+"imageclef.h5")
x_train = tf.constant(x_train/255.0, dtype=tf.float32)
x_test = tf.constant(x_test/255.0, dtype=tf.float32)

BUFFER_SIZE = len(x_train)
BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)
train_data_dist = strategy.experimental_distribute_dataset(train_data)
test_data_dist = strategy.experimental_distribute_dataset(test_data)


with strategy.scope():
    model = myModel()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.build((None, 512, 512, 1))
    model.summary()

    # Define cost function, optimizer and metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    test_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name="test_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracty")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    #@tf.function
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    


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
    

    #@tf.function
    def train_step(images, labels):
        # All ops involving trainables under the GradientTape context manager are recorded for gradient computation purposes
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = compute_loss(labels, predictions)
        
        # Calculate gradients of cost function w.r.t trainable variables and release resources held by GradientTape
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate metrics
        train_accuracy.update_state(labels, predictions)
        return loss

    #@tf.function
    def test_step(images, labels):
        predictions = model(images)
        test_loss.update_state(labels, predictions)
        test_accuracy.update_state(labels, predictions)

    #@tf.function
    def train_step_dist(images, labels):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(images, labels))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    #@tf.function
    def test_step_dist(images, labels):
        return strategy.experimental_run_v2(test_step, args=(images, labels))


    tf.summary.trace_on(graph=True)    

    # Training loop
    step = 0
    total_loss = 0.0
    for epoch in tqdm(range(args.epochs)):
        for images, labels in tqdm(train_data, desc="training"):
            total_loss += train_step_dist(images, labels)
            if step == 0:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="vgg16", step=0)
            step += 1
            train_loss = total_loss / step
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss, step=step)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=step)
                tf.summary.image("image", images, step=step, max_outputs=8)

        for test_images, test_labels in tqdm(test_data, desc="testing"):
            test_step_dist(test_images, test_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
            tf.summary.scalar("val_accuracy", test_accuracy.result(), step=epoch)
            tf.summary.image("val_images", test_images, step=epoch, max_outputs=8)

        ckpt_path = manager.save()
        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {} Test Accuracy: {}, ckpt {}"
        print(template.format(epoch+1, train_loss, tf.round(train_accuracy.result()*100),
                            test_loss.result(), tf.round(test_accuracy.result()*100), ckpt_path))
        
        # Reset metrics for the next epoch
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

