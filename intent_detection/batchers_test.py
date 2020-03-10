"""Tests for batchers.py

Copyright PolyAI Limited.
"""
import unittest

import numpy as np

from intent_detection import batchers


class SamplingBatcherTest(unittest.TestCase):
    def test_rejects_bad_input_labels_not_array(self):
        with self.assertRaises(ValueError):
            batchers.SamplingBatcher(
                examples=np.arange(10),
                labels=list(np.arange(10)),
                batch_size=1
            )

    def test_rejects_bad_input_examples_not_array(self):
        with self.assertRaises(ValueError):
            batchers.SamplingBatcher(
                examples=list(np.arange(10)),
                labels=np.arange(10),
                batch_size=1
            )

    def test_rejects_bad_input_mismatched_dims(self):
        with self.assertRaises(ValueError):
            batchers.SamplingBatcher(
                examples=np.arange(10),
                labels=np.arange(9),
                batch_size=1
            )

    def _test_batcher(self, batch_size, steps, sample_distribution=None):
        np.random.seed(0)
        examples = np.arange(20)
        labels = np.concatenate((
            np.full(
                shape=(5,),
                fill_value=0
            ),
            np.full(
                shape=(5,),
                fill_value=1
            ),
            np.full(
                shape=(5,),
                fill_value=3
            ),
            np.full(
                shape=(5,),
                fill_value=4
            )
        ))
        batcher = batchers.SamplingBatcher(
            examples=examples,
            labels=labels,
            batch_size=batch_size,
            sample_distribution=sample_distribution,
        )

        seen_labels = set()
        for counter, (ex, lab) in enumerate(batcher):
            if counter == steps:
                break
            self.assertEqual(ex.shape, lab.shape)
            self.assertEqual(lab.size, batch_size)

            for x in ex:
                self.assertTrue(x in examples)

            for y in lab:
                seen_labels.add(y)
                self.assertTrue(y in labels)

            for x, y in zip(ex, lab):
                self.assertEqual(labels[x], y)

        self.assertEqual(steps, counter)
        return seen_labels

    def test_batcher_less_classes_than_size(self):
        self._test_batcher(
            batch_size=20,
            steps=5,
        )

    def test_batcher_more_classes_than_size(self):
        self._test_batcher(
            batch_size=3,
            steps=20,
        )

    def test_batcher_zero_weight(self):
        seen_labels = self._test_batcher(
            batch_size=3,
            steps=20,
            sample_distribution={0: 1., 1: 2, 3: 3, 4: 0}
        )
        self.assertNotIn(4, seen_labels)

    def test_batcher_bad_label_in_distribution(self):
        with self.assertRaisesRegex(
                ValueError,
                "label 999 in sample distribution does not exist"):
            self._test_batcher(
                batch_size=3,
                steps=20,
                sample_distribution={0: 1., 1: 2, 3: 3, 999: 0}
            )

    def test_batcher_bad_weight_in_distribution(self):
        with self.assertRaisesRegex(
                ValueError,
                "weight -1 for label 4 is negative"):
            self._test_batcher(
                batch_size=3,
                steps=20,
                sample_distribution={0: 1., 1: 2, 3: 3, 4: -1}
            )


if __name__ == "__main__":
    unittest.main()
