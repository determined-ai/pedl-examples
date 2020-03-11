"""
This example a simple example that shows how to implemented a CNN based on the CIFAR10 in PEDL.

Based off: https://www.tensorflow.org/tutorials/images/cnn
"""

import tensorflow as tf
from tensorflow import keras

import pedl
from pedl.frameworks.keras import data
from pedl.frameworks.keras import TFKerasTrial


def make_data_loaders(experiment_config, hparams):

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    batch_size = pedl.get_hyperparameter("batch_size")
    train = data.InMemorySequence(data=train_images, labels=train_labels, batch_size=batch_size)
    test = data.InMemorySequence(data=test_images, labels=test_labels, batch_size=batch_size)

    return train, test


class MNISTTrial(TFKerasTrial):
    def build_model(self, hparams):
        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model
