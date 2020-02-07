import logging
import os
import tarfile
from typing import Any, Dict

import requests
import tensorflow as tf

WORK_DIRECTORY = "/tmp/pedl-mnist-estimator-work-dir"
MNIST_TF_RECORDS_FILE = "mnist-tfrecord.tar.gz"
MNIST_TF_RECORDS_URL = (
    "https://s3-us-west-2.amazonaws.com/" "determined-ai-test-data/" + MNIST_TF_RECORDS_FILE
)


def download_mnist_tfrecords(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """
    Return the path of a directory with the MNIST dataset in TFRecord format.
    The dataset will be downloaded into WORK_DIRECTORY, if it is not already
    present.
    """
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)

    filepath = os.path.join(WORK_DIRECTORY, MNIST_TF_RECORDS_FILE)
    if not tf.gfile.Exists(filepath):
        logging.info("Downloading {}".format(MNIST_TF_RECORDS_URL))

        r = requests.get(MNIST_TF_RECORDS_URL)
        with tf.gfile.Open(filepath, "wb") as f:
            f.write(r.content)
            logging.info("Downloaded {} ({} bytes)".format(MNIST_TF_RECORDS_FILE, f.size()))

        logging.info("Extracting {} to {}".format(MNIST_TF_RECORDS_FILE, WORK_DIRECTORY))
        with tarfile.open(filepath, mode="r:gz") as f:
            f.extractall(path=WORK_DIRECTORY)

    data_dir = os.path.join(WORK_DIRECTORY, "mnist-tfrecord")
    assert tf.gfile.Exists(data_dir)
    return data_dir
