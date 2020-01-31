# Based on this Keras example model:
#   https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
from typing import Any, Callable, Dict, Tuple

import imdb
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Embedding, LSTM
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import Adam, Optimizer
from keras.preprocessing import sequence
from keras.utils.data_utils import Sequence

from pedl.frameworks.keras import KerasTrial
from pedl.trial import MetricOp, ValidOp


def binary_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - binary_accuracy(y_true, y_pred)


class ImdbLstmTrial(KerasTrial):
    def __init__(self, hparams: Dict):
        self._batch_size = hparams["batch_size"]  # type: int
        self.embedding_size = hparams["embedding_size"]
        self.lstm_output_size = hparams["lstm_output_size"]
        self.dropout = hparams["dropout"]
        self.recurrent_dropout = hparams["recurrent_dropout"]
        self.activation = hparams["activation"]

    def build_model(self, hparams: Dict[str, Any]) -> Sequential:
        model = Sequential()
        model.add(Embedding(20000, self.embedding_size))
        model.add(
            LSTM(
                self.lstm_output_size,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        )
        model.add(Dense(1, activation=self.activation))

        return model

    def optimizer(self) -> Optimizer:
        return Adam()

    def loss(self) -> Callable[[Any, Any], float]:
        loss_func = binary_crossentropy  # type: Callable[[Any, Any], float]
        return loss_func

    def batch_size(self) -> int:
        return self._batch_size

    def training_metrics(self) -> Dict[str, MetricOp]:
        return {"accuracy": binary_accuracy}

    def validation_metrics(self) -> Dict[str, ValidOp]:
        return {"validation_error": binary_error}


class ImdbSequence(Sequence):  # type: ignore
    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int, maxlen: int):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.maxlen = maxlen

    def __len__(self):  # type: ignore
        # Returns number of batches
        return len(self.data) // self.batch_size

    def __getitem__(self, index):  # type: ignore
        # Gets batch at position index
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        x = self.data[start:end]
        x = sequence.pad_sequences(x, maxlen=self.maxlen)
        y = self.labels[start:end]
        return (x, y)


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[Sequence, Sequence]:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=hparams["max_features"])
    train = ImdbSequence(x_train, y_train, hparams["batch_size"], hparams["max_text_len"])
    test = ImdbSequence(x_test, y_test, hparams["batch_size"], hparams["max_text_len"])
    return (train, test)
