import tensorflow as tf


class myModel(tf.keras.Model):

    def __init__(self, input_shape=(256, 256, 1)):
        super().__init__()
        self.input1 = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding="same",activation=tf.nn.relu)
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.flatten = tf.keras.layers.Flatten()
        self.fc6 = tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)   
     
    @tf.function
    def call(self, inputs):
        x = self.input1(inputs)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.flatten(x)

        return self.fc6(x)
