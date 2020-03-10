"""Sentence encoder-based intent classification models

Copyright PolyAI Limited.
"""

import glog
import numpy as np
import tensorflow as tf

from intent_detection.batchers import SamplingBatcher, iter_to_generator


class PolynomialDecay:
    """A callable that implements polynomial decay.

    Used as a callback in keras.
    """
    def __init__(self, max_epochs, init_lr, power=1.0):
        """Creates a new PolynomialDecay

        Args:
            max_epochs: int, maximum number of epochs
            init_lr: float, initial learning rate which will decay
            power: float, the power of the decay function
        """
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.power = power

    def __call__(self, epoch):
        """Calculates the new (smaller) learning rate for the current epoch

        Args:
            epoch: int, the epoch for which we need to calculate the LR

        Returns:
            float, the new learning rate
        """
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_lr * decay

        return float(alpha)


def _train_mlp_with_generator(
        batcher, input_size, steps_per_epoch, label_set, hparams,
        validation_data=None, verbose=1):
    """Trains a Multi Layer Perceptron (MLP) model using keras.

    Args:
        batcher: an instance of a class that inherits from abc.Iterator and
            iterates through batches. see batchers.py for an example.
        input_size: int, length of the input vector
        steps_per_epoch: int, number of steps per one epoch
        label_set: set of ints, the set of labels
        hparams: an instance of tf.contrib.training.Hparams, see config.py
            for some examples
        validation_data: This can be either
            - a generator for the validation data
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
        verbose: keras verbosity mode, 0, 1, or 2.

    Returns:
        keras model, which has been trained
        test accuracy history, as retreived from keras
    """

    hparams.input_size = input_size
    hparams.output_size = len(label_set)

    model = _create_model(hparams)

    callbacks = None
    if hparams.lr_decay_pow:
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(PolynomialDecay(
                max_epochs=hparams.epochs,
                init_lr=hparams.learning_rate,
                power=hparams.lr_decay_pow))]

    glog.info("Training model...")
    history_callback = model.fit_generator(
        generator=iter_to_generator(batcher),
        steps_per_epoch=max(steps_per_epoch, 1),
        epochs=hparams.epochs,
        shuffle=False,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose
    )

    test_acc_history = (None if not validation_data
                        else history_callback.history["val_acc"])

    return model, test_acc_history


def _create_model(hparams):
    model = tf.keras.models.Sequential()
    dropout = hparams.dropout
    optimizer_name = hparams.optimizer
    optimizer = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD
    }[optimizer_name]

    input_size = hparams.input_size
    for _ in range(hparams.num_hidden_layers):
        model.add(
            tf.keras.layers.Dropout(dropout, input_shape=(input_size, ))
        )
        model.add(tf.keras.layers.Dense(hparams.hidden_layer_size,
                                        activation=hparams.activation))
        input_size = hparams.hidden_layer_size

    model.add(tf.keras.layers.Dense(hparams.output_size, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer(lr=hparams.learning_rate),
                  metrics=["accuracy"])
    return model


def train_model(train_encodings, train_labels, categories, hparams,
                validation_data=None, verbose=1):
    """Trains an intent classification model

    Args:
        train_encodings: np.array with the train encodings
        train_labels: list of labels corresponding to each train example
        categories: the set of categories
        hparams: a tf.contrib.training.HParams object containing the model
            and training hyperparameters
        validation_data: (validation_encodings, validation_labels) tuple
        verbose: the keras_model.train() verbose level

    Returns:
        model: a keras model
        eval_acc_history: The evaluation results per epoch

    """
    distribution = None if not hparams.balance_data else {
        x: 1. / len(categories) for x in range(len(categories))}

    batcher = SamplingBatcher(
        train_encodings, train_labels, hparams.batch_size, distribution)

    steps_per_epoch = np.ceil(len(train_labels) / hparams.batch_size)

    model, eval_acc_history = _train_mlp_with_generator(
        batcher, train_encodings.shape[1], steps_per_epoch,
        categories, hparams, validation_data=validation_data, verbose=verbose)
    return model, eval_acc_history
