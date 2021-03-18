import pickle
import numpy as np
import h5py

pathTrainImages = './60_20_20/color_train.pkl'
with open(pathTrainImages, 'rb') as f:
  images, labels = pickle.load(f)

labels = labels[:, None]


def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File('kaggle.hf5', "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images),  data=images
    )
    meta_set = file.create_dataset(
        "labels", np.shape(labels),  data=labels
    )
    file.close()

store_many_hdf5(images, files)