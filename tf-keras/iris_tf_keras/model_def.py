"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in PEDL. Information for how this can be configured can be found in `make_data_loaders()`.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off of: https://medium.com/@nickbortolotti/iris-species-categorization-using-tf-keras-tf-data-and-differences-between-eager-mode-on-and-off-9b4693e0b22
"""
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Sequence, get_file, to_categorical

import pedl
from pedl.frameworks import keras
from pedl.frameworks.keras import TFKerasTensorBoard, TFKerasTrial

# Constants about the data set.
NUM_CLASSES = 3

class IrisTrial(TFKerasTrial):
    def session_config(self, hparams: Dict[str, Any]) -> tf.ConfigProto:
        return tf.ConfigProto()

    def build_model(self, hparams: Dict[str, Any]) -> Sequential:
        model = Sequential()
        model.add(Dense(pedl.get_hyperparameter('layer1_dense_size'), input_dim=4))
        model.add(Dense(NUM_CLASSES))
        model.add(Activation("softmax"))

        model.compile(
            RMSprop(lr=pedl.get_hyperparameter('learning_rate'),
                    decay=pedl.get_hyperparameter('learning_rate_decay')),
            categorical_crossentropy,
            [categorical_accuracy],
        )

        return model

    def keras_callbacks(self, hparams: Dict[str, Any]) -> List[tf.keras.callbacks.Callback]:
        return [TFKerasTensorBoard(update_freq="batch", profile_batch=0, histogram_freq=1)]

def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[Sequence, Sequence]:
    """
    Provides training and validation data for model training.
    """
    download_dir = pedl.get_download_data_dir()
    return get_data(download_dir)

def get_data(data_path: str) -> Tuple[Sequence, Sequence]:
    ds_columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Plants']
    species = np.array(['Setosa', 'Versicolor', 'Virginica'], dtype=np.object)

    # Load data
    categories = 'Plants'
    train = pd.read_csv(os.path.join(data_path, "train.csv"), names=ds_columns, header=0)
    train_features, train_labels = train, train.pop(categories)
    test = pd.read_csv(os.path.join(data_path, "test.csv"), names=ds_columns, header=0)
    test_features, test_labels = test, test.pop(categories)

    train_labels_categorical = to_categorical(train_labels, num_classes=3)
    test_labels_categorical = to_categorical(test_labels, num_classes=3)

    train = keras.data.InMemorySequence(
        data=train_features,
        labels=train_labels_categorical,
        batch_size=pedl.get_hyperparameter('batch_size')
    )
    test = keras.data.InMemorySequence(
        data=test_features,
        labels=test_labels_categorical,
        batch_size=pedl.get_hyperparameter('batch_size')
    )
    return train, test

def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """
    Downloads training and validation data for model training.
    """
    subdirectory = "iris_dataset"
    path_train = get_file("train.csv", origin=experiment_config["data"]["train_url"], cache_subdir=subdirectory)  # type: str
    path_test = get_file("test.csv", origin=experiment_config["data"]["test_url"], cache_subdir=subdirectory)  # type: str
    return os.path.dirname(os.path.abspath(path_test))
