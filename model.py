import os
import csv
import tensorflow as tf
import numpy as np

from scipy import ndimage

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

import data_generator as DataGenerator

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 100, "The batch size.")

fine_tune = False

# Data
# Import data
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data.pop(0)

# print(data[0][0], print(data[0][3]))

# Samples
print("# Samples: %d" % len(data))

images = [data[i][0] \
          for i in range(len(data))]
angles = [data[i][3] \
              for i in range(len(data))]

# Check data
print("# Images: %d" % len(images))
print("# Angles: %d" % len(angles))
print("# Image 0: ", images[0])
print("# Angle 0: ", angles[0])

# Numpy arrays
images = np.array(images)
angles = np.array(angles).astype(np.float32)

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1, random_state=42)

# Check train-val datasets shapes
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)

## Build model

# Model 1
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'))
# model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
# model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, name='output', activation='tanh'))

# Model 2
# model = Sequential()
# model.add(Lambda(lambda x: x/127.5 - 1.,
#         input_shape=(80, 40, 3),
#         output_shape=(80, 40, 3)))
# model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(512))
# model.add(Dropout(.5))
# model.add(ELU())
# model.add(Dense(1))

def get_model():
    # Model 3
    model = Sequential()
    model.add(Lambda(lambda x: x/128 -1.,
                       input_shape=(200, 66, 3),
                       output_shape=(200, 66, 3)))

    # CNN Layer 1
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 2
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 3
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 4
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
    # CNN Layer 5
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # FCNN Layer 1
    model.add(Dense(100, activation='relu'))
    # FCNN Layer 2
    model.add(Dense(50, activation='relu'))
    # FCNN Layer 3
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    return model

if fine_tune:
    learning_rate = 0.001
    print('From Trained Model')
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
else:
    learning_rate = 0.01
    model = get_model()


# Optimizer
optimizer = Adam(lr=learning_rate)

# Compile model
model.compile(optimizer=optimizer, loss='mse')

model.fit_generator(
    DataGenerator.get_batch(X_train, y_train, FLAGS.batch_size),
    samples_per_epoch=DataGenerator.get_samples_per_epoch(X_train.shape[0], FLAGS.batch_size),
    nb_epoch=FLAGS.epochs,
    validation_data=DataGenerator.get_batch(X_train, y_train, FLAGS.batch_size),
    nb_val_samples=DataGenerator.get_samples_per_epoch(X_val.shape[0], FLAGS.batch_size)
  )

## Extract model data
# Save weights
model.save_weights("model.h5")

# Save model config (architecture)
json = model.to_json()
with open("model.json", "w") as f:
    f.write(json)
