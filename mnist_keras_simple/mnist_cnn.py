"""Trains a simple convnet on the MNIST dataset."""
import logging

import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import pedl

BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
IMG_ROWS, IMG_COLS = INPUT_SHAPE[0], INPUT_SHAPE[0]


def cnn_model():
    # Get hyperparameters for this trial.
    kernel_size = pedl.get_hyperparameter("kernel_size")
    dropout = pedl.get_hyperparameter("dropout")
    activation = pedl.get_hyperparameter("activation")

    model = Sequential()
    model.add(
        Conv2D(
            32, kernel_size=(kernel_size, kernel_size), activation="relu", input_shape=INPUT_SHAPE
        )
    )
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return model


def load_mnist_data():
    # Download and prepare the MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    logging.info(
        "Training MNIST in PEDL, experiment {}, trial {}".format(
            pedl.get_experiment_id(), pedl.get_trial_id()
        )
    )
    experiment_cfg = pedl.get_experiment_config()
    logging.info("Experiment configuration: ", experiment_cfg)

    model = cnn_model()
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Create training and test data generators using Keras' ImageDataGenerator.
    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    val_datagen = ImageDataGenerator()

    # Compute quantities required for featurewise normalization.
    train_datagen.fit(x_train)

    # Additional configuration of input queueing from the 'data' section in the
    # experiment configuration.
    data = experiment_cfg.get("data", {})
    use_multiprocessing = data.get("use_multiprocessing", False)
    workers = data.get("workers", 1)
    max_queue_size = data.get("max_queue_size", 10)

    model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE),
        validation_steps=y_test.shape[0] // BATCH_SIZE,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        max_queue_size=max_queue_size,
    )
