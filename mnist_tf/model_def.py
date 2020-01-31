"""Trains a simple convnet on the MNIST dataset using TensorFlow."""
from typing import Any, Dict

import tensorflow
from packaging import version

from pedl.frameworks.tensorflow import TensorFlowTrial
from pedl.util import get_experiment_config

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


DATA_SET_SIZE = 60000
NUM_CLASSES = 10


class MNistTrial(TensorFlowTrial):
    def __init__(self, hparams: Dict[str, Any]):
        # Hyperparameters used by the optimizer.
        self.base_learning_rate = hparams["base_learning_rate"]
        self.my_batch_size = hparams["batch_size"]  # type: int
        self.weight_cost = hparams["weight_cost"]

        # Hyperparameters that influence the model architecture.
        self.n_filters1 = hparams["n_filters1"]
        self.n_filters2 = hparams["n_filters2"]

        # Hyperparameters can also be optionally specified with a default:
        self.disable_CPU_parallelism = hparams.get("disable_CPU_parallelism", False)
        self._training_metrics = hparams.get(
            "training_metrics", ["learning_rate", "error"]
        )  # type: list
        self._validation_metrics = hparams.get("validation_metrics", ["error"])  # type: list

    def training_metrics(self) -> list:
        return self._training_metrics

    def validation_metrics(self) -> list:
        return self._validation_metrics

    def build_graph(self, record: Any, is_training: tf.Tensor) -> dict:
        """The Model definition."""
        conv1 = tf.layers.conv2d(
            inputs=tf.cast(record["image"], tf.float32),
            filters=self.n_filters1,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.n_filters2,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_shape = pool2.get_shape().as_list()

        pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
        dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=is_training)

        logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

        # Training computation: logits + cross-entropy loss.
        labels = record["label"]
        training_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

        predictions = tf.argmax(logits, axis=1)
        correct = tf.cast(tf.equal(predictions, labels), tf.float32)
        accuracy = tf.reduce_mean(correct)
        error = 1 - accuracy

        training_set_size = DATA_SET_SIZE - get_experiment_config()["data"]["validation_set_size"]
        self.learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,
            self.training_counter(),
            training_set_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True,
        )

        avg_prediction = tf.reduce_mean(tf.one_hot(predictions, NUM_CLASSES), axis=0)

        return {
            "avg_prediction": avg_prediction,
            "learning_rate": self.learning_rate,
            "loss": training_loss,
            "error": error,
        }

    def optimizer(self) -> tf.train.Optimizer:
        return tf.train.MomentumOptimizer(self.learning_rate, 0.9)

    def session_config(self) -> tf.ConfigProto:
        # If disable CPU parallelism is true, disable multithreaded TensorFlow
        # CPU operations to achieve floating point reproducibility at the cost
        # of performance. Otherwise, use the default configuration.
        if self.disable_CPU_parallelism:
            return tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        else:
            return super().session_config()
