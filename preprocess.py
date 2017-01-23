import os
import csv

import pickle
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

# Import data
with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data.pop(0)

# print(data[0][0], print(data[0][3]))

# Samples
print("# Samples: %d" % len(data))

images_paths = [data[i][0] \
          for i in range(len(data))]
angles = [data[i][3] \
              for i in range(len(data))]

# Check data
print("# Images: %d" % len(images_paths))
print("# Angles: %d" % len(angles))

# Import images
images = np.array([cv2.imread('data/'+image_path) for image_path in images_paths])

# Numpy arrays
images = np.array(images).astype(np.float32)[:100]
angles = np.array(angles).astype(np.float32)[:100]

# Check image shape (320x160)
print("NP Images ", images.shape)
print("NP Image shape: ", images[0].shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.2, random_state=42)

# Check train-test datasets shapes
print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

# Export data
if True:
  pickle_file = './data/driving.p'

  try:
    fp = open(pickle_file, 'wb')
    save = {
      'X_train': X_train,
      'y_train': y_train,
      'X_test': X_test,
      'y_test': y_test
      }
    pickle.dump(save, fp, pickle.HIGHEST_PROTOCOL)
    fp.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)
