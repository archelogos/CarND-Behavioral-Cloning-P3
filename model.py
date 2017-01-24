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

flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 100, "The batch size.")

# Getting metadata
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data.pop(0)

# Position 0 for center images, position 3 for steering angles
# print(data[0][0], print(data[0][3]))

# How many samples are in this training phase
print("# Samples: %d" % len(data))

# Getting data from the indexes mentioned before
images = [data[i][0] \
          for i in range(len(data))]
angles = [data[i][3] \
              for i in range(len(data))]

# Checking if it has consistency and we are taking the correct element
print("# Images: %d" % len(images))
print("# Angles: %d" % len(angles))
print("# Image 0: ", images[0])
print("# Angle 0: ", angles[0])

# Transforming them to Numpy arrays
images = np.array(images)
angles = np.array(angles).astype(np.float32)

# Train-Validation split
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1, random_state=42)

# Check Train and Validation dataset shapes
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)

# Model definition (NVIDIA model from the paper 5-CNN + 3 FCNN)
def get_model():
    model = Sequential()
    # Regularization technique from comma.ai paper
    model.add(Lambda(lambda x: x/128 -1.,
                       input_shape=(200, 66, 3),
                       output_shape=(200, 66, 3)))

    # CNN Layer 1 (x2 subsampling in the 3 first conv nets)
    # relu instead or elu or tanh activation function helps to converge faster
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 2
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 3
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    # CNN Layer 4
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
    # CNN Layer 5
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
    # 0.4 Dropout layer to prevent too much overfitting (the model is overfitted anyway)
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # FCNN Layer 1
    model.add(Dense(100, activation='relu'))
    # FCNN Layer 2
    model.add(Dense(50, activation='relu'))
    # FCNN Layer 3
    model.add(Dense(10, activation='relu'))
    # Regression-Like Output (1 Class)
    model.add(Dense(1))

    return model

try:
    # If it's not the first training, we are loading model from the saved file
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
    print('From Trained Model')
except:
    model = get_model()

# As it happened in the previous project with the Traffic Signs, to choose
# Adam Optimizer is one of the key parts in this practice.
optimizer = Adam()

# Summarizing the model just to check that everything is fine
model.summary()

# We are defining the mean squared error as the loss function.
# Altough it doesn't represent perfectly the direction of the
# training phase, at least it helps us to know if we are doing really bad.
# If we have a poor mse it will be a good idea to change the
# strategy. However, having a very low mse doesn't mean that the car can
# complete a lap properly
# Mixing up some concepts from Reinforcement Learning and
# this model could be better approach to this problem.
model.compile(optimizer=optimizer, loss='mse')

# We are generating dynamically the batches from the data that we loaded before
# Instead of preprocess everything in a previous phase, we are preprocessing it
# in "real time" and passing it to the model through the generator
model.fit_generator(
    DataGenerator.get_batch(X_train, y_train, FLAGS.batch_size),
    samples_per_epoch=DataGenerator.get_samples_per_epoch(X_train.shape[0], FLAGS.batch_size),
    nb_epoch=FLAGS.epochs,
    validation_data=DataGenerator.get_batch(X_train, y_train, FLAGS.batch_size),
    nb_val_samples=DataGenerator.get_samples_per_epoch(X_val.shape[0], FLAGS.batch_size)
  )

# Saving weights
model.save_weights("model.h5")

# Saving the model
json = model.to_json()
with open("model.json", "w") as f:
    f.write(json)
