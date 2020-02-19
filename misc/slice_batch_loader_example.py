"""
Example using SliceBatchLoader for CIFAR-10.

Download and unpack the CIFAR-10 data set by running these shell commands:

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz

Description from https://www.cs.toronto.edu/~kriz/cifar.html:

The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images
dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey
Hinton. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10
classes, with 6000 images per class. There are 50000 training images and 10000
test images.

The dataset is divided into five training batches and one test batch, each with
10000 images. The test batch contains exactly 1000 randomly-selected images
from each class. The training batches contain the remaining images in random
order, but some training batches may contain more images from one class than
another. Between them, the training batches contain exactly 5000 images from
each class.
"""

import glob
import os
import pickle

import numpy as np

import pedl.data

IMAGE_SIZE = 32 * 32 * 3


def make_data_loaders(experiment_config, hparams):
    """
    Loads all CIFAR-10 data into a single in-memory ArrayBatchLoader, and
    return training and validation slices using SliceBatchLoader.  The path
    where the CIFAR-10 pickle files are located should be in
    experiment_config["data"]["path"].
    """
    # Load and concatenate all of the data.
    data = np.empty((0, IMAGE_SIZE), dtype="uint8")
    labels = []
    for pkl in sorted(glob.glob(os.path.join(experiment_config["data"]["path"], "*_batch*"))):
        records = pickle.load(open(pkl, "rb"), encoding="bytes")
        data = np.concatenate([data, records[b"data"]])
        labels.extend(records[b"labels"])

    data_loader = pedl.data.ArrayBatchLoader(data, np.array(labels))

    # Split into training and validation slices.
    training_loader = pedl.data.SliceBatchLoader(data_loader, 0, 50000)
    validation_loader = pedl.data.SliceBatchLoader(data_loader, 50000, 10000)

    return training_loader, validation_loader
