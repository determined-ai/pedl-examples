"""
Train a single layer neural network on MNIST with two classification outputs.
The first output and loss function is the 10-class classification that is
normally used to train MNIST. The second output and loss function is a binary
classification target that predicts if the input digit is >= 5.
"""

import logging
from typing import Any, Callable, Dict, Tuple

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils.data_utils import Sequence
from mnist_utils import (
    DATA_SET_SIZE,
    download,
    extract_data,
    extract_labels,
    INPUT_SHAPE,
    NUM_CLASSES,
)

import pedl
from pedl.frameworks.keras import KerasTrial
from pedl.trial import MetricOp, ValidOp


class MNISTTrial(KerasTrial):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        self._batch_size = hparams["batch_size"]  # type: int
        self._learning_rate = hparams["learning_rate"]  # type: float
        self._dropout = hparams["dropout"]  # type: float
        self._activation = hparams["activation"]
        self._kernel_size = hparams["kernel_size"]
        self._momentum = hparams.get("momentum", 0.9)

    def build_model(self, hparams: Dict[str, Any]) -> Model:
        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=(self._kernel_size, self._kernel_size),
                activation="relu",
                input_shape=INPUT_SHAPE,
            )
        )
        model.add(Conv2D(64, (3, 3), activation=self._activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self._dropout))
        model.add(Flatten())
        model.add(Dense(128, activation=self._activation))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation="softmax"))
        return model

    def batch_size(self) -> int:
        return self._batch_size

    def optimizer(self) -> keras.optimizers.Optimizer:
        return SGD(lr=self._learning_rate, momentum=self._momentum)

    def loss(self) -> Callable[[Any, Any], float]:
        loss_func = categorical_crossentropy  # type: Callable[[Any, Any], float]
        return loss_func

    def training_metrics(self) -> Dict[str, MetricOp]:
        return {"accuracy": categorical_accuracy}

    def validation_metrics(self) -> Dict[str, ValidOp]:
        return {"accuracy": categorical_accuracy}


class MNISTSequence(Sequence):  # type: ignore
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        assert len(X) == len(y)
        self._X = X
        self._y = y
        self._batch_size = batch_size

    def __len__(self) -> int:
        return len(self._X) // self._batch_size

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        start = index * self._batch_size
        end = (index + 1) * self._batch_size
        labels = keras.utils.to_categorical(self._y[start:end], NUM_CLASSES)

        return (self._X[start:end], labels)


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[MNISTSequence, MNISTSequence]:
    data_config = experiment_config["data"]
    base_url = data_config["base_url"]
    training_data_file = data_config["training_data"]
    training_labels_file = data_config["training_labels"]
    validation_set_size = data_config["validation_set_size"]
    batch_size = hparams["batch_size"]

    if not pedl.is_valid_url(base_url):
        raise ValueError("Invalid base_url: {}".format(base_url))

    assert DATA_SET_SIZE > validation_set_size
    training_set_size = DATA_SET_SIZE - validation_set_size

    tmp_data_file = download(base_url, training_data_file)
    tmp_labels_file = download(base_url, training_labels_file)

    train_data = extract_data(tmp_data_file, DATA_SET_SIZE)
    train_labels = extract_labels(tmp_labels_file, DATA_SET_SIZE)

    # Shuffle the data.
    indices = np.arange(DATA_SET_SIZE)
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Generate the validation set.
    validation_loader = MNISTSequence(
        train_data[:validation_set_size, ...], train_labels[:validation_set_size], batch_size
    )

    training_loader = MNISTSequence(
        train_data[validation_set_size:, ...], train_labels[validation_set_size:], batch_size
    )

    assert len(training_loader) == training_set_size // batch_size
    assert len(validation_loader) == validation_set_size // batch_size

    logging.info(
        "Extracted MNIST data: {} training batches, {} validation batches".format(
            training_set_size, validation_set_size
        )
    )

    return training_loader, validation_loader
