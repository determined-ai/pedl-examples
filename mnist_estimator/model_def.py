"""
Trains a simple DNN on the MNIST dataset using the TensorFlow Estimator API.
"""
import os
from typing import Any, Callable, Dict, List, Tuple

import tensorflow
from packaging import version

import pedl
from pedl import get_download_data_dir
from pedl.frameworks.tensorflow import EstimatorTrial, wrap_dataset, wrap_optimizer

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

IMAGE_SIZE = 28
NUM_CLASSES = 10


def parse_mnist_tfrecord(serialized_example: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Parse a TFRecord representing a single MNIST data point into an input
    feature tensor and a label tensor.

    Returns: (features: Dict[str, Tensor], label: Tensor)
    """
    raw = tf.parse_example(
        serialized=serialized_example, features={"image_raw": tf.FixedLenFeature([], tf.string)}
    )
    image = tf.decode_raw(raw["image_raw"], tf.float32)

    label_dict = tf.parse_example(
        serialized=serialized_example, features={"label": tf.FixedLenFeature(1, tf.int64)}
    )
    return {"image": image}, label_dict["label"]


class MNistTrial(EstimatorTrial):
    def build_estimator(self, hparams: Dict[str, Any]) -> tf.estimator.Estimator:
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams["learning_rate"])
        # Call `wrap_optimizer` immediately after creating your optimizer.
        optimizer = wrap_optimizer(optimizer)
        return tf.estimator.DNNClassifier(
            feature_columns=[
                tf.feature_column.numeric_column(
                    "image", shape=(IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32
                )
            ],
            n_classes=NUM_CLASSES,
            hidden_units=[
                hparams["hidden_layer_1"],
                hparams["hidden_layer_2"],
                hparams["hidden_layer_3"],
            ],
            config=tf.estimator.RunConfig(tf_random_seed=pedl.get_trial_seed()),
            optimizer=optimizer,
            dropout=hparams["dropout"],
        )

    def _input_fn(
        self, hparams: Dict[str, Any], files: List[str], shuffle_and_repeat: bool = False
    ) -> Callable:
        def _fn() -> tf.data.TFRecordDataset:
            dataset = tf.data.TFRecordDataset(files)
            # Call `wrap_dataset` immediately after creating your dataset.
            dataset = wrap_dataset(dataset)
            if shuffle_and_repeat:
                dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(1000))
            dataset = dataset.batch(hparams["batch_size"])
            dataset = dataset.map(parse_mnist_tfrecord)
            return dataset

        return _fn

    @staticmethod
    def _get_filenames(directory: str) -> List[str]:
        return [os.path.join(directory, path) for path in tf.gfile.ListDirectory(directory)]

    def build_train_spec(self, hparams: Dict[str, Any]) -> tf.estimator.TrainSpec:
        download_data_dir = get_download_data_dir()
        train_files = self._get_filenames(os.path.join(download_data_dir, "train"))
        return tf.estimator.TrainSpec(self._input_fn(hparams, train_files, shuffle_and_repeat=True))

    def build_validation_spec(self, hparams: Dict[str, Any]) -> tf.estimator.EvalSpec:
        download_data_dir = get_download_data_dir()
        val_files = self._get_filenames(os.path.join(download_data_dir, "validation"))
        return tf.estimator.EvalSpec(self._input_fn(hparams, val_files, shuffle_and_repeat=False))
