"""Train an intent classifier based on a sentence encoder and interact with it

usage:
    python run_classifier_interactive.py --train_file <INTENT_FILE_PATH> \
  --params config.default \
  --params_overrides encoder_type=${ENC}

<INTENT_FILE_PATH> must point to CSV file with the following header:
    text,category

Copyright PolyAI Limited.
"""

import argparse
import csv

import numpy as np
import tensorflow as tf

from intent_detection.classifier import train_model
from intent_detection.encoder_clients import get_encoder_client
from intent_detection.utils import _object_from_name


def _preprocess_data(encoder_client, train_file):
    """Reads the data from the files, encodes it and parses the labels

    Args:
        encoder_client: an EncoderClient
        train_file: The absolute path to the training data

    Returns:
        categories, encodings, labels

    """
    with tf.gfile.Open(train_file, "r") as data_file:
        data = np.array(list(csv.reader(data_file))[1:])
        labels = data[:, 1]
        encodings = encoder_client.encode_sentences(data[:, 0])

    categories = np.unique(labels)

    # convert labels to integers
    labels = np.array([np.argwhere(categories == x)[0][0] for x in labels])

    return categories, encodings, labels


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file', required=True,
        help="The absolute path to the training data")
    parser.add_argument(
        "--params",
        default="intent_detection.config.default",
        help="Path to config file containing the model hyperparameters")
    parser.add_argument(
        "--params_overrides",
        help="A comma-separated list of param=value pairs specifying overrides"
             " to the model hyperparameters.", default="")

    parsed_args = parser.parse_args()
    hparams = _object_from_name(parsed_args.params)
    hparams.parse(parsed_args.params_overrides)

    encoder_client = get_encoder_client(
        hparams.encoder_type, cache_dir=hparams.cache_dir
    )
    categories, encodings, labels = _preprocess_data(
        encoder_client, parsed_args.train_file
    )

    print(f"Your labels are here, make sure they are correct: {categories}")

    model, _ = train_model(
        encodings, labels, categories, hparams)

    print("Now you will be able to speak to the model. Press Ctrl + C to quit")
    while True:
        query = input("Your query:")
        query_encoding = encoder_client.encode_sentences([query])
        output = model.predict(query_encoding).flatten()
        prediction = np.argmax(output)
        print(f"Prediction: {categories[prediction]}, "
              f"score: {output[prediction]}")


if __name__ == "__main__":
    _main()
