import pickle
import tensorflow as tf
import numpy as np

from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


pickle_file = './data/driving.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  X_train = save['X_train']
  y_train = save['y_train']
  X_test = save['X_test']
  y_test = save['y_test']
  del save  # hint to help gc free up memory
  print('Training set', X_train.shape, y_train.shape)
  print('Test set', X_test.shape, y_test.shape)
  print('Validation set', X_validation.shape, y_validation.shape)


## Build model
# @TODO

# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_split=0.1, shuffle=True)


## Extract model data
# Save weights
model.save_weights("model.h5")

# Save model config (architecture)
json_string = model.to_json()
with open("model.json", "w") as f:
    f.write(json_string)
