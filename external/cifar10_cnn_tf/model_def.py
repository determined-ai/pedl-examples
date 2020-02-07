"""Port of the cifar10_cnn_keras example to TensorFlow."""
from typing import Any, Dict, List

import tensorflow
from packaging import version

from pedl.frameworks.tensorflow import TensorFlowTrial

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10


class CIFARTrial(TensorFlowTrial):
    def __init__(self, hparams: Dict[str, Any]):
        self.base_learning_rate = hparams["learning_rate"]
        self.learning_rate_decay = hparams["learning_rate_decay"]
        self.layer1_dropout = hparams["layer1_dropout"]
        self.layer2_dropout = hparams["layer2_dropout"]
        self.layer3_dropout = hparams["layer3_dropout"]
        self.my_batch_size = hparams["batch_size"]  # type: int
        self.disable_cpu_parallelism = hparams.get("disable_CPU_parallelism", False)

    def session_config(self) -> tf.ConfigProto:
        if self.disable_cpu_parallelism:
            return tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        return tf.ConfigProto(allow_soft_placement=True)

    def batch_size(self) -> int:
        return self.my_batch_size

    def training_metrics(self) -> List[str]:
        return ["error"]

    def validation_metrics(self) -> List[str]:
        return ["error"]

    def build_graph(self, record: tf.Tensor, is_training: tf.Tensor) -> dict:
        self.model_vars = self.define_vars()

        logits = self.model_arch(tf.cast(record["image"], tf.float32), self.model_vars, is_training)
        training_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=record["label"], logits=logits)
        )

        predictions = tf.argmax(logits, axis=1)
        correct = tf.cast(tf.equal(predictions, record["label"]), tf.float32)
        accuracy = tf.reduce_mean(correct)
        error = 1 - accuracy

        return {"loss": training_loss, "error": error}

    def optimizer(self) -> tf.train.Optimizer:
        # Keras appears to use `inverse_time_decay`, 1 step per batch.
        self.learning_rate = tf.train.inverse_time_decay(
            self.base_learning_rate,
            self.training_counter(),
            1,  # Decay step.
            self.learning_rate_decay,
        )  # Decay rate.

        # Keras rho defaults to 0.9
        # Keras rho == tf decay
        # Keras epsilon defaults to 1e-7
        # Keras doesn't have tf momemtum, so set to zero
        return tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, epsilon=1e-7, momentum=0.0)

    def define_vars(self) -> dict:
        return {
            "conv1_weights": tf.get_variable(
                name="conv1_weights",
                dtype=tf.float32,
                shape=[3, 3, NUM_CHANNELS, 32],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "conv1_biases": tf.get_variable(
                name="conv1_biases",
                dtype=tf.float32,
                shape=[32],
                initializer=tf.zeros_initializer(),
            ),
            "conv2_weights": tf.get_variable(
                name="conv2_weights",
                dtype=tf.float32,
                shape=[3, 3, 32, 32],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "conv2_biases": tf.get_variable(
                name="conv2_biases",
                dtype=tf.float32,
                shape=[32],
                initializer=tf.zeros_initializer(),
            ),
            "conv3_weights": tf.get_variable(
                name="conv3_weights",
                dtype=tf.float32,
                shape=[3, 3, 32, 64],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "conv3_biases": tf.get_variable(
                name="conv3_biases",
                dtype=tf.float32,
                shape=[64],
                initializer=tf.zeros_initializer(),
            ),
            "conv4_weights": tf.get_variable(
                name="conv4_weights",
                dtype=tf.float32,
                shape=[3, 3, 64, 64],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "conv4_biases": tf.get_variable(
                name="conv4_biases",
                dtype=tf.float32,
                shape=[64],
                initializer=tf.zeros_initializer(),
            ),
            "fc1_weights": tf.get_variable(
                name="fc1_weights",
                dtype=tf.float32,
                shape=[(((IMAGE_SIZE - 2) // 2 - 2) // 2) ** 2 * 64, 512],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "fc1_biases": tf.get_variable(
                name="fc1_biases", dtype=tf.float32, shape=[512], initializer=tf.zeros_initializer()
            ),
            "fc2_weights": tf.get_variable(
                name="fc2_weights",
                dtype=tf.float32,
                shape=[512, NUM_CLASSES],
                initializer=tf.glorot_uniform_initializer(),
            ),
            "fc2_biases": tf.get_variable(
                name="fc2_biases",
                dtype=tf.float32,
                shape=[NUM_CLASSES],
                initializer=tf.zeros_initializer(),
            ),
        }

    def model_arch(self, data_node: tf.Tensor, vars: dict, is_training: tf.Tensor) -> tf.Tensor:
        """The Model definition."""
        conv1 = tf.nn.conv2d(data_node, vars["conv1_weights"], strides=[1, 1, 1, 1], padding="SAME")

        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, vars["conv1_biases"]))

        conv2 = tf.nn.conv2d(relu1, vars["conv2_weights"], strides=[1, 1, 1, 1], padding="VALID")

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, vars["conv2_biases"]))

        pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        drop1 = tf.layers.dropout(pool1, self.layer1_dropout, training=is_training)

        conv3 = tf.nn.conv2d(drop1, vars["conv3_weights"], strides=[1, 1, 1, 1], padding="SAME")

        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, vars["conv3_biases"]))

        conv4 = tf.nn.conv2d(relu3, vars["conv4_weights"], strides=[1, 1, 1, 1], padding="VALID")

        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, vars["conv4_biases"]))

        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        drop2 = tf.layers.dropout(pool2, self.layer2_dropout, training=is_training)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool2_shape = drop2.get_shape().as_list()

        flatten = tf.reshape(drop2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])

        dense1 = tf.matmul(flatten, vars["fc1_weights"]) + vars["fc1_biases"]

        relu5 = tf.nn.relu(dense1)

        drop3 = tf.layers.dropout(relu5, self.layer3_dropout, training=is_training)

        return tf.matmul(drop3, vars["fc2_weights"]) + vars["fc2_biases"]
