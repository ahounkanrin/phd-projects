import tensorflow as tf

INPUT_SIZE = (200, 200)
# Define the model
baseModel = tf.keras.applications.InceptionV3(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), 
                                              include_top=False, weights="imagenet")
input_cnn = baseModel.input
input_hog = tf.keras.Input(shape=(4356,)) # | (20736,)  | (900,)

#x = baseModel.output
features_cnn = tf.keras.layers.GlobalAveragePooling2D()(baseModel.output)
features_hog = tf.keras.layers.Flatten()(input_hog)
x = tf.keras.layers.Concatenate([features_cnn, features_hog])
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.2)(x)
outputs = tf.keras.layers.Dense(360, activation="softmax")(x)
model = tf.keras.Model(inputs=[input_cnn, input_hog], outputs=outputs)
model.summary()

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


# <pre>mixed10 (Concatenate)           (None, 4, 4, 2048)   0           activation_85[0][0]
#                                                                  mixed9_1[0][0]
#                                                                  concatenate_1[0][0]
#                                                                  activation_93[0][0]
# __________________________________________________________________________________________________
# global_average_pooling2d (Globa (None, 2048)         0           mixed10[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 1024)         2098176     global_average_pooling2d[0][0]
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 1024)         0           dense[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 1024)         1049600     dropout[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 1024)         0           dense_1[0][0]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 360)          369000      dropout_1[0][0]
# ==================================================================================================
# Total params: 25,319,560
# Trainable params: 25,285,128
# Non-trainable params: 34,432
