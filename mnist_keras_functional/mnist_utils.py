import gzip
import logging
import os
import urllib.parse

import numpy as np
import requests
import tensorflow as tf

# Constants about the data set.
DATA_SET_SIZE = 60000
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
PIXEL_DEPTH = 255


def download(base_url: str, filename: str) -> str:
    work_directory = "/tmp/work_dir"

    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory, filename)
    if not tf.gfile.Exists(filepath):
        url = urllib.parse.urljoin(base_url, filename)
        logging.info("Downloading {}".format(url))

        r = requests.get(url, stream=True)
        with tf.gfile.Open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

            logging.info("Downloaded {} ({} bytes)".format(filename, f.size()))

    return filepath


def extract_data(path: str, num_images: int) -> np.ndarray:
    """
    Extract the images into a 4D tensor [image index, y, x, channels]. Values
    are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    with tf.gfile.Open(path, "rb") as zip_f:
        with gzip.GzipFile(fileobj=zip_f) as f:
            f.read(16)
            buf = f.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            return data


def extract_labels(path: str, num_images: int) -> np.ndarray:
    """Extract the labels into a vector of int64 label IDs."""
    with tf.gfile.Open(path, "rb") as zip_f:
        with gzip.GzipFile(fileobj=zip_f) as f:
            f.read(8)
            buf = f.read(1 * num_images)
            return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
