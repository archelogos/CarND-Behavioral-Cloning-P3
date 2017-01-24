import math
import numpy as np
from scipy.misc import imread, imresize

def read_images(image_paths):
    images = np.empty([image_paths.shape[0], 160, 320, 3])

    for i, path in enumerate(image_paths):
        images[i] = imread('data/'+path)

    return images


def preprocess(images):
    shape = (200, 66, 3)
    height, width, channels = shape
    images_resized = np.empty([images.shape[0], height, width, channels])
    for i, img in enumerate(images):
        images_resized[i] = imresize(img, shape)

    images = images_resized

    return images


def augment(images, angles):
    new_images = np.empty_like(images)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(images, angles)):
        if np.random.choice(2):
            new_images[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_images[i] = img
            new_angles[i] = angle

    images = new_images
    angles = new_angles

    return images, angles


def get_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


def get_batch(images, angles, batch_size):

    samples = len(images)

    while True:
        selected = np.random.choice(samples, batch_size)
        images_batch, angles_batch = read_images(images[selected]), angles[selected].astype(float)

        images_batch, angles_batch = augment(preprocess(images_batch), angles_batch)

        yield images_batch, angles_batch
