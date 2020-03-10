"""This module contains batch iterators that are used in EncoderClassifiers

Copyright PolyAI Limited.
"""
from collections import abc
from typing import Dict, Optional

import numpy as np

_MAX_PER_BATCH = 3


class SamplingBatcher(abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """
    def __init__(
            self,
            examples: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            sample_distribution: Optional[Dict[int, float]] = None,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples: np.ndarray containing examples
            labels: np.ndarray containing labels
            batch_size: int size of a single batch
            sample_distribution: optional distribution over label
                classes for sampling. This is normalized to sum to 1. Defines
                the target distribution that batches will be sampled with.
                Defaults to the data distribution.
        """
        _validate_labels_examples(examples, labels)
        self._examples = examples
        self._labels = labels
        self._label_classes = np.unique(labels)
        self._class_to_indices = {
            label: np.argwhere(labels == label).flatten()
            for label in self._label_classes
        }
        if sample_distribution is None:
            # Default to the data distribution
            sample_distribution = {
                label: float(indices.size)
                for label, indices in self._class_to_indices.items()
            }
        self._label_choices, self._label_probs = (
            self._get_label_choices_and_probs(sample_distribution))
        self._batch_size = batch_size

    def _get_label_choices_and_probs(self, sample_distribution):
        label_choices = []
        label_probs = []
        weight_sum = sum(sample_distribution.values())
        for label, weight in sample_distribution.items():
            if label not in self._labels:
                raise ValueError(
                    f"label {label} in sample distribution does not exist")
            if weight < 0.0:
                raise ValueError(
                    f"weight {weight} for label {label} is negative")
            label_choices.append(label)
            label_probs.append(weight / weight_sum)

        return np.array(label_choices), np.array(label_probs)

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        class_choices = np.random.choice(
            self._label_choices, size=self._batch_size, p=self._label_probs)

        batch_indices = []
        for class_choice in class_choices:
            indices = self._class_to_indices[class_choice]
            batch_indices.append(np.random.choice(indices))

        return self._examples[batch_indices], self._labels[batch_indices]

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self


def _validate_labels_examples(examples, labels):
    if not isinstance(examples, np.ndarray):
        raise ValueError("examples should be an ndarray")

    if not isinstance(labels, np.ndarray):
        raise ValueError("labels should be ndarray")

    if not labels.size == examples.shape[0]:
        raise ValueError("number of labels != number of examples")


def iter_to_generator(iterator):
    """Gets a generator from an iterator.

    Used so that keras type checking does not complain.

    Args:
        iterator: any python iterator

    Returns:
        a python generator that just calls next on the iterator
    """

    def gen():
        while True:
            yield next(iterator)

    return gen()
