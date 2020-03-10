"""Python utils

Copyright PolyAI Limited.
"""

import argparse
import importlib


def _object_from_name(object_name):
    """Creates an object from its name as a python string.

    Args:
        object_name: the name of the object to be created as a string.

    Returns: the object that the string refers to, dynamically imported.
    """
    module_name, top_name = object_name.rsplit('.', 1)
    the_module = importlib.import_module(module_name)
    return getattr(the_module, top_name)


def parse_args_and_hparams():
    """Parses the command line arguments and hparams

    Returns:
        parsed_args: The parsed comand line arguments
        hparams: tf.contrib.training.HParams object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', required=True,
        help="Path to the folder containing the data")

    parser.add_argument(
        '--output_dir', required=True,
        help="Path to the folder where the output file withe the results will "
             "be saved")

    parser.add_argument(
        "--params", default="intent_detection.config.default",
        help="Path to config file containing the model hyperparameters")

    parser.add_argument(
        "--params_overrides",
        help="A comma-separated list of param=value pairs specifying overrides"
             " to the model hyperparameters.", default="")

    parsed_args = parser.parse_args()
    hparams = _object_from_name(parsed_args.params)
    hparams.parse(parsed_args.params_overrides)

    return parsed_args, hparams
