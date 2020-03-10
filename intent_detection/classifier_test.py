"""Tests for classifier.py

Copyright PolyAI Limited.
"""

import unittest

import numpy as np
import tensorflow as tf

from intent_detection import classifier


class PolynomialDecayTest(unittest.TestCase):
    def test_simple_usage(self):
        init_lr = 0.5
        decay_obj = classifier.PolynomialDecay(
            max_epochs=5,
            init_lr=init_lr,
            power=1.5
        )

        self.assertEqual(decay_obj(0), init_lr)
        self.assertAlmostEqual(decay_obj(1), 0.35777, places=5)
        self.assertAlmostEqual(decay_obj(4), 0.04472, places=5)
        self.assertEqual(decay_obj(5), 0)


class TrainModelTest(unittest.TestCase):
    test_hparams = tf.contrib.training.HParams(
        # model hparams
        epochs=2,
        learning_rate=0.7,
        lr_decay_pow=1,
        batch_size=3,
        num_hidden_layers=1,
        hidden_layer_size=8,
        activation="relu",
        dropout=0.75,
        optimizer="sgd",
        encoder_type="convert",

        # training hparams
        balance_data=True,
    )

    def test_training_no_validation(self):
        training_examples = np.array([[1, 2, 3], [3, 2, 1],
                                      [4, 5, 6], [6, 5, 4]])
        training_labels = np.array([0, 1, 0, 1])
        label_set = {0, 1}

        model, acc_hist = classifier.train_model(
            train_encodings=training_examples,
            train_labels=training_labels,
            categories=label_set,
            hparams=TrainModelTest.test_hparams
        )

        self.assertIsNone(acc_hist)
        self.assertIsInstance(model, tf.keras.models.Sequential)

        pred = model(np.array([[10.5, 20, 30]]))
        self.assertEqual(pred.shape, (1, 2))

    def test_training_validation(self):
        training_examples = np.array([[1, 2, 3], [3, 2, 1],
                                      [4, 5, 6], [6, 5, 4]])
        training_labels = np.array([0, 1, 0, 1])
        label_set = {0, 1}

        model, acc_hist = classifier.train_model(
            train_encodings=training_examples,
            train_labels=training_labels,
            categories=label_set,
            hparams=TrainModelTest.test_hparams,
            validation_data=(
                np.array([[10, 20, 30], [30, 20, 10]]), np.array([0, 1])
            )
        )

        self.assertEqual(len(acc_hist), 2)


if __name__ == "__main__":
    unittest.main()
