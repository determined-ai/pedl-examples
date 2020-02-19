from typing import Any, Dict, Tuple

import tensorflow
from packaging import version

import tensorflow_datasets as tfds

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    mnist = tfds.image.MNIST()
    mnist.download_and_prepare()
    datasets = mnist.as_dataset(shuffle_files=hparams.get("shuffle_files", True))
    batch_size = hparams["batch_size"]
    return (datasets["train"].batch(batch_size), datasets["test"].batch(batch_size))
