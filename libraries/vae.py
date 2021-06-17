# Import Tensorflow 2.0
import tensorflow as tf

import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import pickle
import h5py


# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl

# Get the training data: both images from CelebA and ImageNet
path_to_training_data = tf.keras.utils.get_file('train_face.h5', 'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
# Instantiate a TrainingDatasetLoader using the downloaded dataset
loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data)

with open('/content/TrainImages/60_20_20/color_train.pkl', 'rb') as f:
  imagechallenge, _ = pickle.load(f)

imagechallenge = [cv2.resize(x, (64, 64))[None] for x in imagechallenge]
imagechallenge = np.vstack(imagechallenge)
imagechallenge = np.asarray(imagechallenge*255, dtype=np.uint8)

images, labels = loader.images[...,::-1], loader.labels

print(f'[INFO] Original image dataset shape : {images.shape}, \n[INFO] Kagglechallenge data shape: {imagechallenge.shape}')

# if we just want faces and labels of kaggle challenge
#images = images[np.where(labels == 0)[0]] 
#labels = labels[np.where(labels== 0)[0]] 

l = len(imagechallenge)
images = np.concatenate([images , imagechallenge], axis = 0)
del imagechallenge
labels = np.concatenate([labels, np.ones((l, 1))], axis = 0)

print(f'[INFO] Cancatenated images and labels shape: {images.shape}, {labels.shape}')

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File('kaggle.h5', "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images),  data=images
    )
    meta_set = file.create_dataset(
        "labels", np.shape(labels),  data=labels
    )
    file.close()

store_many_hdf5(images[...,::-1], labels)
