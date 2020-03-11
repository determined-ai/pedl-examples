"""
This example a simple example that shows how to implemented a CNN based on the CIFAR10 in PEDL.

Based off: https://www.tensorflow.org/tutorials/images/cnn
"""
from typing import Any, Dict, Tuple

import numpy as np
from tensorflow.keras import datasets, layers, models, utils

from pedl.frameworks import keras
from pedl.frameworks.keras import TFKerasTrial


def preprocess_data(data: np.ndarray) -> np.ndarray:
    return data / 255.0


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    return utils.to_categorical(labels)


class CIFARTrial(TFKerasTrial):
    def build_model(self, hparams: Dict[str, Any]) -> models.Sequential:
        """
        Builds a simple CNN model.
        """
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
        )

        return model


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[utils.Sequence, utils.Sequence]:
    """
    Provides training and validation data for model training.
    """

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    batch_size = hparams["batch_size"]

    # TODO Replace keras.data.InMemorySequence with something simpler
    train = keras.data.InMemorySequence(
        data=preprocess_data(train_images),
        labels=preprocess_labels(train_labels),
        batch_size=batch_size,
    )

    test = keras.data.InMemorySequence(
        data=preprocess_data(test_images),
        labels=preprocess_labels(test_labels),
        batch_size=batch_size,
    )
    return train, test
