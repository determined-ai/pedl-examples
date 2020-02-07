"""
Example custom batch loader for CIFAR-10.

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

import functools
import glob
import os
import pickle

import numpy as np

import pedl.data


IMAGE_SIZE = 32 * 32 * 3


class CIFAR10BatchLoader(pedl.data.BatchLoader):
    def __init__(self, pickle_files, file_length=10000):
        """
        pickle_files is the list of input pickle files.  file_length is the
        number of records in each pickle file (defaults to 10,000).
        """
        self.pickle_files = pickle_files
        self.file_length = file_length

    def __len__(self):
        return len(self.pickle_files) * self.file_length

    @functools.lru_cache(maxsize=2)
    def load_file(self, idx):
        """
        Load one of the pickle files, by index.  This is cached (via
        `functools.lru_cache`) to minimize I/O.
        """
        assert 0 <= idx < len(self.pickle_files)
        return pickle.load(open(self.pickle_files[idx], "rb"), encoding="bytes")

    def get_batch(self, start, end):
        """
        Reads a Batch from the CIFAR-10 pickle files.
        """
        assert 0 <= start <= end <= len(self)
        start_file, start_off = divmod(start, self.file_length)
        end_file, end_off = divmod(end - 1, self.file_length)
        data = np.empty((0, IMAGE_SIZE), dtype="uint8")
        labels = []
        # Walk through the pickle files and accumulate the requested records.
        while start_file <= end_file:
            records = self.load_file(start_file)
            file_end_off = end_off + 1 if end_file == start_file else self.file_length
            data = np.concatenate([data, records[b"data"][start_off:file_end_off, :]])
            labels.extend(records[b"labels"][start_off:file_end_off])
            start_file, start_off = start_file + 1, 0
        return pedl.data.Batch(data, np.array(labels))


def make_data_loaders(experiment_config, hparams):
    """
    Returns training and validation CIFAR10BatchLoaders.  The path where the
    CIFAR-10 pickle files are located should be in
    experiment_config["data"]["path"].
    """
    training_loader = CIFAR10BatchLoader(
        sorted(glob.glob(os.path.join(experiment_config["data"]["path"], "data_batch_*")))
    )
    validation_loader = CIFAR10BatchLoader(
        [os.path.join(experiment_config["data"]["path"], "test_batch")]
    )
    return training_loader, validation_loader
