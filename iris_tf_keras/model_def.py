"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in PEDL. Information for how this can be configured can be found in `make_data_loaders()`.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off of: https://medium.com/@nickbortolotti/iris-species-categorization-using-tf-keras-tf-data-
              and-differences-between-eager-mode-on-and-off-9b4693e0b22
"""
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

import pedl
from pedl.frameworks.keras import TFKerasTensorBoard, TFKerasTrial

# Constants about the data set.
NUM_CLASSES = 3


class IrisTrial(TFKerasTrial):
    def build_model(self, hparams: Dict[str, Any]) -> Model:
        """
        Define model for iris classification.

        This is a simple model with one hidden layer to predict iris species (setosa, versicolor, or
        virginica) based on four input features (length and width of sepals and petals).
        """
        inputs = Input(shape=(4,))
        dense1 = Dense(pedl.get_hyperparameter("layer1_dense_size"))(inputs)
        dense2 = Dense(NUM_CLASSES, activation="softmax")(dense1)
        model = Model(inputs=inputs, outputs=dense2)

        model.compile(
            RMSprop(
                lr=pedl.get_hyperparameter("learning_rate"),
                decay=pedl.get_hyperparameter("learning_rate_decay"),
            ),
            categorical_crossentropy,
            [categorical_accuracy],
        )

        return model

    def keras_callbacks(self, hparams: Dict[str, Any]) -> List[tf.keras.callbacks.Callback]:
        return [TFKerasTensorBoard(update_freq="batch", profile_batch=0, histogram_freq=1)]
