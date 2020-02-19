"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in PEDL. Information for how this can be configured can be found in `make_data_loaders()`.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Sequence, get_file

import tensorflow_datasets as tfds
from cifar_utils import NUM_CLASSES, augment_data, get_data, preprocess_data, preprocess_labels
from pedl import get_download_data_dir
from pedl.frameworks import keras
from pedl.frameworks.keras import TFKerasTensorBoard, TFKerasTrial, wrap_dataset
from pedl.frameworks.keras.data import KerasDataAdapter

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3


def categorical_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - categorical_accuracy(y_true, y_pred)  # type: ignore


class CIFARTrial(TFKerasTrial):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        self.base_learning_rate = hparams["learning_rate"]  # type: float
        self.learning_rate_decay = hparams["learning_rate_decay"]  # type: float
        self.layer1_dropout = hparams["layer1_dropout"]  # type: float
        self.layer2_dropout = hparams["layer2_dropout"]  # type: float
        self.layer3_dropout = hparams["layer3_dropout"]  # type: float

    def session_config(self, hparams: Dict[str, Any]) -> tf.ConfigProto:
        if hparams.get("disable_CPU_parallelism", False):
            return tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        else:
            return tf.ConfigProto()

    def build_model(self, hparams: Dict[str, Any]) -> Sequential:
        model = Sequential()
        model.add(tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="image"))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.layer1_dropout))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.layer2_dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(self.layer3_dropout))
        model.add(Dense(NUM_CLASSES, name="label"))
        model.add(Activation("softmax"))

        model.compile(
            RMSprop(lr=self.base_learning_rate, decay=self.learning_rate_decay),
            categorical_crossentropy,
            [categorical_accuracy, categorical_error],
        )

        return model

    def keras_callbacks(self, hparams: Dict[str, Any]) -> List[tf.keras.callbacks.Callback]:
        return [TFKerasTensorBoard(update_freq="batch", profile_batch=0, histogram_freq=1)]


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[Sequence, Sequence]:
    """
    Provides training and validation data for model training.

    This example demonstrates how you could configure PEDL to help you optimize your data loading.
    """

    is_tf_dataset = experiment_config.get("data", {}).get("use_tf_dataset", False)
    if is_tf_dataset:
        return create_cifar10_tf_dataset(experiment_config, hparams)
    else:
        return create_cifar10_sequence(experiment_config, hparams)


def create_cifar10_sequence(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[Sequence, Sequence]:
    """
    In this example we added some fields of note under the `data` field in the YAML experiment
    configuration: the `acceleration` field. Under this field, you can configure multithreading by
    setting `use_multiprocessing` to `False`, or set it to `True` for multiprocessing. You can also
    configure the number of workers (processes or threads depending on `use_multiprocessing`).

    Another thing of note are the data augmentation fields in hyperparameters. The fields here get
    passed through to Keras' `ImageDataGenerator` for real-time data augmentation.
    """
    acceleration = experiment_config["data"].get("acceleration")
    width_shift_range = hparams.get("width_shift_range", 0.0)
    height_shift_range = hparams.get("height_shift_range", 0.0)
    horizontal_flip = hparams.get("horizontal_flip", False)
    batch_size = hparams["batch_size"]

    download_dir = get_download_data_dir()
    (train_data, train_labels), (test_data, test_labels) = get_data(download_dir)

    # Setup training data loader.
    data_augmentation = {
        "width_shift_range": width_shift_range,
        "height_shift_range": height_shift_range,
        "horizontal_flip": horizontal_flip,
    }
    train = augment_data(train_data, train_labels, batch_size, data_augmentation)

    if acceleration:
        workers = acceleration.get("workers", 1)
        use_multiprocessing = acceleration.get("use_multiprocessing", False)
        train = KerasDataAdapter(train, workers=workers, use_multiprocessing=use_multiprocessing)

    # Setup validation data loader.
    test = keras.data.InMemorySequence(
        data=preprocess_data(test_data),
        labels=preprocess_labels(test_labels),
        batch_size=batch_size,
    )

    return train, test


def create_cifar10_tf_dataset(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    download_dir = get_download_data_dir()

    cifar10 = tfds.image.Cifar10(data_dir=download_dir)
    cifar10.download_and_prepare(download_dir=download_dir)
    datasets = cifar10.as_dataset()
    ds_train, ds_test = datasets["train"], datasets["test"]

    ds_train = wrap_dataset(ds_train)
    ds_test = wrap_dataset(ds_test)

    batch_size = hparams["batch_size"]
    ds_train = ds_train.map(
        lambda x: (tf.divide(x["image"], 255), tf.one_hot(x["label"], NUM_CLASSES))
    )
    ds_test = ds_test.map(
        lambda x: (tf.divide(x["image"], 255), tf.one_hot(x["label"], NUM_CLASSES))
    )

    return ds_train.batch(batch_size), ds_test.batch(batch_size)


def download_cifar10_tf_dataset() -> str:
    dirname = "cifar-10-tf-dataset"
    cifar10 = tfds.image.Cifar10(data_dir=dirname)
    cifar10.download_and_prepare(download_dir=dirname)
    return dirname


def download_cifar10_tf_sequence(url: str) -> str:
    dirname = "cifar-10-batches-py"
    # This is copied from keras.datasets.cifar10 and modified to support
    # a custom origin URL
    path = get_file(dirname, origin=url, untar=True)  # type: str
    return path


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """
    Downloads training and validation data for model training.
    """
    is_tf_dataset = experiment_config.get("data", {}).get("use_tf_dataset", False)
    if is_tf_dataset:
        return download_cifar10_tf_dataset()
    else:
        return download_cifar10_tf_sequence(url=experiment_config["data"]["url"])
