"""Train and evaluate an intent classifier based on a sentence encoder

usage:
    python run_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.default \
  --output_dir=<OUTPUT_DIR> \
  --params_overrides task=${DS},data_regime=${DR},encoder_type=${ENC}

Copyright PolyAI Limited.
"""

import csv
import json
import os
import pandas as pd
import glog
import numpy as np
import tensorflow as tf

from intent_detection.classifier import train_model
from intent_detection.encoder_clients import get_encoder_client
from intent_detection.utils import parse_args_and_hparams

_TRAIN = "train"
_TEST = "test"


def _preprocess_data(encoder_client, hparams, data_dir):
    """Reads the data from the files, encodes it and parses the labels

    Args:
        encoder_client: an EncoderClient
        hparams: a tf.contrib.training.HParams object containing the model
            and training hyperparameters
        data_dir: The directory where the inten data has been downloaded

    Returns:
        categories, encodings, labels

    """
    labels = {}
    encodings = {}
    if hparams.task=='wallet':
        test = pd.read_csv('/polyai-models/polyai-models/200909_test_huawei_wallet_30strat_min3_manyanswers.csv')
        train = pd.read_csv('/polyai-models/polyai-models/200909_train_huawei_wallet_60strat_min3_manyanswers.csv')
        val = pd.read_csv('/polyai-models/polyai-models/200909_val_huawei_wallet_10strat_min3_manyanswers.csv')
        train = train.append(val)
        print('WALLET TRAIN SHAPE',train.shape)
        categories = train.labels.unique()
        labels[_TRAIN]=train.labels.values
        encodings[_TRAIN] = encoder_client.encode_sentences(train.text.values)
        labels[_TEST]=test.labels.values
        encodings[_TEST] = encoder_client.encode_sentences(test.text.values)
    else:
        if hparams.data_regime == "full":
            train_file = "train"
        elif hparams.data_regime == "10":
            train_file = "train_10"
        elif hparams.data_regime == "30":
            train_file = "train_30"
        else:
            glog.error(f"Invalid data regime: {hparams.data_regime}")
        train_data = os.path.join(
            data_dir, hparams.task, f"{train_file}.csv")
        test_data = os.path.join(data_dir, hparams.task, "test.csv")
        categories_file = os.path.join(data_dir, hparams.task, "categories.json")

        with tf.gfile.Open(categories_file, "r") as categories_file:
            categories = json.load(categories_file)



        with tf.gfile.Open(train_data, "r") as data_file:
            data = np.array(list(csv.reader(data_file))[1:])
            labels[_TRAIN] = data[:, 1]
            encodings[_TRAIN] = encoder_client.encode_sentences(data[:, 0])

        with tf.gfile.Open(test_data, "r") as data_file:
            data = np.array(list(csv.reader(data_file))[1:])
            labels[_TEST] = data[:, 1]
            encodings[_TEST] = encoder_client.encode_sentences(data[:, 0])

        # convert labels to integers
        labels = {
            k: np.array(
                [categories.index(x) for x in v]) for k, v in labels.items()
        }

    return categories, encodings, labels


def _main():
    parsed_args, hparams = parse_args_and_hparams()

    if hparams.task.lower() not in ["clinc", "hwu", "banking", "wallet"]:
        raise ValueError(f"{hparams.task} is not a valid task")

    encoder_client = get_encoder_client(hparams.encoder_type,
                                        cache_dir=hparams.cache_dir)

    categories, encodings, labels = _preprocess_data(
        encoder_client, hparams, parsed_args.data_dir)

    accs = []
    eval_acc_histories = []
    if hparams.eval_each_epoch:
        validation_data = (encodings[_TEST], labels[_TEST])
        verbose = 1
    else:
        validation_data = None
        verbose = 0

    for seed in range(hparams.seeds):
        glog.info(f"### Seed {seed} ###")
        model, eval_acc_history = train_model(
            encodings[_TRAIN], labels[_TRAIN], categories, hparams,
            validation_data=validation_data, verbose=verbose)

        _, acc = model.evaluate(encodings[_TEST], labels[_TEST], verbose=0)
        glog.info(f"Seed accuracy: {acc:.3f}")
        accs.append(acc)
        eval_acc_histories.append(eval_acc_history)

    average_acc = np.mean(accs)
    variance = np.std(accs)
    glog.info(
        f"Average results:\n"
        f"Accuracy: {average_acc:.3f}\n"
        f"Variance: {variance:.3f}")

    results = {
        "Average results": {
            "Accuracy": float(average_acc),
            "Variance": float(variance)
        }
    }
    if hparams.eval_each_epoch:
        results["Results per epoch"] = [
            [float(x) for x in y] for y in eval_acc_histories]

    if not tf.gfile.Exists(parsed_args.output_dir):
        tf.gfile.MakeDirs(parsed_args.output_dir)
    with tf.gfile.Open(
            os.path.join(parsed_args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    _main()
