from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

import tensorpack as tp
from pedl import get_download_data_dir
from pedl.frameworks.tensorflow.tensorpack_trial import TensorpackTrial
from tensorpack.callbacks import TFEventWriter

IMAGE_SIZE = 28


class Model(tp.ModelDesc):  # type: ignore
    def __init__(self, hparams: Dict[str, Any]) -> None:
        self.hparams = hparams

    def inputs(self) -> List[tf.TensorSpec]:
        """
        Define all the inputs (shape, data_type, name) that the graph will need.
        """
        return [
            tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE), tf.float32, "input"),
            tf.TensorSpec((None,), tf.int32, "label"),
        ]

    def build_graph(self, image: Any, label: Any) -> Any:
        """
        This function builds the model which takes the input
        variables and returns cost.
        """

        # In tensorflow, inputs to convolution function are assumed to be NHWC.
        # Add a single channel here.
        image = tf.expand_dims(image, 3)

        # Center the pixels values at zero.
        image = image * 2 - 1

        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3.
        with tp.argscope(
            tp.Conv2D, kernel_size=3, activation=tf.nn.relu, filters=self.hparams["n_filters"]
        ):
            logits = (
                tp.LinearWrap(image)
                .Conv2D("conv0")
                .MaxPooling("pool0", 2)
                .Conv2D("conv1")
                .MaxPooling("pool1", 2)
                .FullyConnected("fc0", 512, activation=tf.nn.relu)
                .Dropout("dropout", rate=0.5)
                .FullyConnected("fc1", 10, activation=tf.identity)()
            )

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name="cross_entropy_loss")  # the average cross-entropy loss

        correct = tf.cast(
            tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name="correct"
        )
        accuracy = tf.reduce_mean(correct, name="accuracy")
        train_error = tf.reduce_mean(1 - correct, name="train_error")
        tp.summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers.
        wd_cost = tf.multiply(
            self.hparams["weight_cost"],
            tp.regularize_cost("fc.*/W", tf.nn.l2_loss),
            name="regularize_loss",
        )
        total_cost = tf.add_n([wd_cost, cost], name="loss")

        return total_cost

    def optimizer(self) -> Any:
        lr = tf.train.exponential_decay(
            learning_rate=self.hparams["base_learning_rate"],
            global_step=tp.get_global_step_var(),
            decay_steps=self.hparams["decay_steps"],
            decay_rate=self.hparams["decay_rate"],
            staircase=True,
            name="learning_rate",
        )
        tf.summary.scalar("lr", lr)
        return tf.train.AdamOptimizer(lr)


class MNISTTrial(TensorpackTrial):
    def build_model(self, hparams: Dict[str, Any], trainer_type: str) -> tp.ModelDesc:
        return Model(hparams)

    def training_metrics(self, hparams: Dict[str, Any]) -> List[str]:
        return ["learning_rate"]

    def validation_metrics(self, hparams: Dict[str, Any]) -> List[str]:
        return ["cross_entropy_loss", "accuracy"]

    def tensorpack_monitors(self, hparams: Dict[str, Any]) -> List[tp.MonitorBase]:
        return [TFEventWriter()]


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[Optional[tp.DataFlow], Optional[tp.DataFlow]]:
    """Provides training and validation data for model training."""
    download_dir = get_download_data_dir()
    training_dataflow = tp.BatchData(
        tp.dataset.Mnist("train", dir=download_dir), hparams["batch_size"]
    )
    validation_dataflow = tp.BatchData(
        tp.dataset.Mnist("test", dir=download_dir), hparams["batch_size"]
    )

    return training_dataflow, validation_dataflow


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """Downloads training and validation data for model training."""
    download_dir = "mnist_data"
    _ = tp.dataset.Mnist("train", dir=download_dir)
    _ = tp.dataset.Mnist("test", dir=download_dir)
    return download_dir
