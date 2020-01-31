"""
Train a single layer neural network on MNIST with two classification
outputs. The first output and loss function is the 10-class classification
that is normally used to train MNIST. The second output and loss function is
a binary classification target that predicts if the input digit is >= 5.
"""

import logging
from typing import Any, Dict, Tuple

import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.data_utils import Sequence
from mnist_utils import (
    DATA_SET_SIZE,
    download,
    extract_data,
    extract_labels,
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES,
)

import pedl
from pedl.frameworks.keras import KerasFunctionalTrial


class MultiTaskMNistTrial(KerasFunctionalTrial):
    def __init__(self, hparams: Dict[str, Any]):
        self._batch_size = hparams["batch_size"]  # type: int
        self.learning_rate = hparams["learning_rate"]  # type: float
        self.layer_size = hparams["layer_size"]  # type: int
        self.dropout = hparams["dropout"]  # type: float

    def build_model(self, hparams: Dict[str, Any]) -> Model:
        x = inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name="input")
        x = Flatten()(x)
        x = Dense(self.layer_size, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        digit_predictions = Dense(NUM_CLASSES, activation="softmax", name="digit_predictions")(x)
        binary_predictions = Dense(1, activation="sigmoid", name="binary_predictions")(x)
        return Model(inputs=inputs, outputs=[digit_predictions, binary_predictions])

    def batch_size(self) -> int:
        return self._batch_size

    def optimizer(self) -> keras.optimizers.Optimizer:
        return SGD(lr=self.learning_rate)

    def losses(self) -> dict:
        return {
            "binary_predictions": binary_crossentropy,
            "digit_predictions": categorical_crossentropy,
        }

    def training_metrics(self) -> dict:
        return {
            "binary_accuracy": ("binary_predictions", binary_accuracy),
            "digit_accuracy": ("digit_predictions", categorical_accuracy),
        }

    def validation_metrics(self) -> dict:
        return {
            "binary_accuracy": ("binary_predictions", binary_accuracy),
            "digit_accuracy": ("digit_predictions", categorical_accuracy),
        }


class MultiTaskMNISTSequence(Sequence):  # type: ignore
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.X) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        digit_labels = keras.utils.to_categorical(self.y[start:end], NUM_CLASSES)
        binary_labels = np.where(self.y[start:end] >= 5, 1, 0)

        return (
            {"input": self.X[start:end]},
            {"binary_predictions": binary_labels, "digit_predictions": digit_labels},
        )


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[MultiTaskMNISTSequence, MultiTaskMNISTSequence]:
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
    validation_loader = MultiTaskMNISTSequence(
        train_data[:validation_set_size, ...], train_labels[:validation_set_size], batch_size
    )

    training_loader = MultiTaskMNISTSequence(
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
