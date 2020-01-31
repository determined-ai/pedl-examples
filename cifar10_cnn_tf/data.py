from typing import Any, Dict, Tuple

import tensorflow
import tensorflow_datasets as tfds
from packaging import version

# Handle TensorFlow compatibility issues.
if version.parse(tensorflow.__version__) >= version.parse("1.14.0"):
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    cifar10 = tfds.image.Cifar10()
    cifar10.download_and_prepare()
    datasets = cifar10.as_dataset(shuffle_files=hparams.get("shuffle_files", True))
    batch_size = hparams["batch_size"]
    return (datasets["train"].batch(batch_size), datasets["test"].batch(batch_size))
