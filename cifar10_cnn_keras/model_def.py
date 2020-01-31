"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in PEDL. Information for how this can be configured can be found in `make_data_loaders()`.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from cifar_utils import augment_data, CIFARSequence, get_data, NUM_CLASSES
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.optimizers import Optimizer, RMSprop
from keras.utils.data_utils import Sequence

from pedl.frameworks.keras import KerasTrial
from pedl.frameworks.keras.data import KerasDataAdapter
from pedl.trial import MetricOp


# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3


def categorical_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - categorical_accuracy(y_true, y_pred)  # type: ignore


class CIFARTrial(KerasTrial):
    def __init__(self, hparams: Dict[str, Any]):
        self.base_learning_rate = hparams["learning_rate"]  # type: float
        self.learning_rate_decay = hparams["learning_rate_decay"]  # type: float
        self.layer1_dropout = hparams["layer1_dropout"]  # type: float
        self.layer2_dropout = hparams["layer2_dropout"]  # type: float
        self.layer3_dropout = hparams["layer3_dropout"]  # type: float
        self._batch_size = hparams["batch_size"]  # type: int

    def session_config(self, hparams: Dict[str, Any]) -> tf.ConfigProto:
        if hparams.get("disable_CPU_parallelism", False):
            return tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        else:
            return tf.ConfigProto()

    def build_model(self, hparams: Dict[str, Any]) -> Sequential:
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), padding="same", input_shape=[IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        )
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
        model.add(Dense(NUM_CLASSES))
        model.add(Activation("softmax"))

        return model

    def optimizer(self) -> Optimizer:
        return RMSprop(lr=self.base_learning_rate, decay=self.learning_rate_decay)

    def loss(self) -> MetricOp:
        return categorical_crossentropy  # type: ignore

    def batch_size(self) -> int:
        return self._batch_size

    def training_metrics(self) -> Dict[str, Any]:
        return {"accuracy": categorical_accuracy}

    def validation_metrics(self) -> Dict[str, Any]:
        return {"validation_error": categorical_error}


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

    Another thing of note are the data augmentaiton fields in hyperparameters. The fields here get
    passed through to Keras' `ImageDataGenerator` for real-time data augmentation.
    """

    acceleration = experiment_config["data"].get("acceleration")
    url = experiment_config["data"]["url"]
    width_shift_range = hparams.get("width_shift_range", 0.0)
    height_shift_range = hparams.get("height_shift_range", 0.0)
    horizontal_flip = hparams.get("horizontal_flip", False)
    batch_size = hparams["batch_size"]

    (train_data, train_labels), (test_data, test_labels) = get_data(url)

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
    test = CIFARSequence(test_data, test_labels, batch_size)

    return train, test
