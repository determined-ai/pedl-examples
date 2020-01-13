"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in PEDL. Information for how this can be configured can be found in `make_data_loaders()`.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, get_file, to_categorical
from tensorflow.python.keras.datasets.cifar import load_batch

import pedl
from pedl.frameworks import keras
from pedl.frameworks.keras import TFKerasTensorBoard, TFKerasTrial
from pedl.frameworks.keras.data import KerasDataAdapter

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10


class CIFARTrial(TFKerasTrial):
    def session_config(self, hparams: Dict[str, Any]) -> tf.ConfigProto:
        return tf.ConfigProto()

    def build_model(self, hparams: Dict[str, Any]) -> Sequential:
        model = Sequential()
        model.add(tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="image"))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(pedl.get_hyperparameter('layer1_dropout')))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(pedl.get_hyperparameter('layer2_dropout')))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(pedl.get_hyperparameter('layer3_dropout')))
        model.add(Dense(NUM_CLASSES, name="label"))
        model.add(Activation("softmax"))

        model.compile(
            RMSprop(lr=pedl.get_hyperparameter('learning_rate'),
                    decay=pedl.get_hyperparameter('learning_rate_decay')),
            categorical_crossentropy,
            [categorical_accuracy],
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
    In this example we added some fields of note under the `data` field in the YAML experiment
    configuration: the `acceleration` field. Under this field, you can configure multithreading by
    setting `use_multiprocessing` to `False`, or set it to `True` for multiprocessing. You can also
    configure the number of workers (processes or threads depending on `use_multiprocessing`).

    Another thing of note are the data augmentation fields in hyperparameters. The fields here get
    passed through to Keras' `ImageDataGenerator` for real-time data augmentation.
    """
    width_shift_range = pedl.get_hyperparameter("width_shift_range")
    height_shift_range = pedl.get_hyperparameter("height_shift_range")
    horizontal_flip = pedl.get_hyperparameter("horizontal_flip")
    batch_size = pedl.get_hyperparameter("batch_size")

    download_dir = pedl.get_download_data_dir()
    (train_data, train_labels), (test_data, test_labels) = get_data(download_dir)


    # Setup training data loader.
    data_augmentation = {
        "width_shift_range": width_shift_range,
        "height_shift_range": height_shift_range,
        "horizontal_flip": horizontal_flip,
    }
    train = augment_data(train_data, train_labels, batch_size, data_augmentation)
    train = KerasDataAdapter(train, workers=1, use_multiprocessing=True)

    # Setup validation data loader.
    test = keras.data.InMemorySequence(
        data=preprocess_data(test_data),
        labels=preprocess_labels(test_labels),
        batch_size=batch_size,
    )

    return train, test


def preprocess_data(data: np.ndarray) -> np.ndarray:
    return data.astype("float32") / 255


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    return to_categorical(labels, NUM_CLASSES)


def augment_data(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    data_augmentation: Dict[str, Any],
    shuffle: bool = False,
) -> Sequence:
    datagen = ImageDataGenerator(**data_augmentation)
    data = preprocess_data(data)
    labels = preprocess_labels(labels)
    return datagen.flow(data, labels, batch_size=batch_size, shuffle=shuffle)


def get_data(data_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    num_train_samples = 50000

    train_data = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    train_labels = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(data_path, "data_batch_" + str(i))
        (
            train_data[(i - 1) * 10000 : i * 10000, :, :, :],
            train_labels[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(data_path, "test_batch")
    test_data, test_labels = load_batch(fpath)

    train_labels = np.reshape(train_labels, (len(train_labels), 1))
    test_labels = np.reshape(test_labels, (len(test_labels), 1))

    if tf.keras.backend.image_data_format() == "channels_last":
        train_data = train_data.transpose(0, 2, 3, 1)
        test_data = test_data.transpose(0, 2, 3, 1)

    return (train_data, train_labels), (test_data, test_labels)


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """
    Downloads training and validation data for model training.
    """

    dirname = "cifar-10-batches-py"
    # This is copied from keras.datasets.cifar10 and modified to support
    # a custom origin URL
    path = get_file(dirname, origin=experiment_config["data"]["url"], untar=True)  # type: str
    return path
