"""Train and evaluate an intent classifier based on a finetuned BERT model

https://arxiv.org/abs/1810.04805

This code has been adapted from
https://github.com/google-research/bert/blob/master/run_classifier.py

usage:
    python run_bert_finetuned_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.bert_ft \
  --output_dir=<OUTPUT_DIR>

Copyright PolyAI Limited.
"""

import csv
import json
import os

import tensorflow as tf
from bert import modeling, tokenization
from bert.run_classifier import (DataProcessor, InputExample,
                                 PaddingInputExample,
                                 file_based_convert_examples_to_features,
                                 model_fn_builder)

from intent_detection.utils import parse_args_and_hparams

_EVAL_BATCH_SIZE = 8


class _IntentProcessor(DataProcessor):
    """Processor for intent detection data sets."""

    def __init__(self, data_dir, task, train_set):
        """constructor"""
        self._data_dir = data_dir
        self._task = task
        self._train_set = train_set

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(
                os.path.join(data_dir, self._task, f"{self._train_set}.csv"),
                delimiter=",", quotechar='"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(
                os.path.join(data_dir, self._task, "test.csv"), delimiter=",",
                quotechar='"'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(
                os.path.join(data_dir, self._task, "test.csv"), delimiter=",",
                quotechar='"'), "test")

    def get_labels(self):
        """See base class."""
        with tf.gfile.Open(os.path.join(
                self._data_dir, self._task, "categories.json"), "r") as f:
            categories = json.load(f)
        return categories

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a coma separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def _file_based_input_fn_builder(input_file, seq_length, is_training,
                                 drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.

    This is duplicated from the original file in bert.run_classifier to
    increase the buffer_size of d.shuffle, otherwise the training won't
    converge"""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports
        # tf.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=15000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def main(_):  # NOQA
    parsed_args, hparams = parse_args_and_hparams()
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(hparams.bert_config_file)

    tf.gfile.MakeDirs(parsed_args.output_dir)

    if hparams.data_regime == "full":
        train_file = "train"
    elif hparams.data_regime == "10":
        train_file = "train_10"
    elif hparams.data_regime == "30":
        train_file = "train_30"
    else:
        tf.logging.error(f"Invalid data regime: {hparams.data_regime}")

    processor = _IntentProcessor(parsed_args.data_dir, hparams.task,
                                 train_file)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=hparams.vocab_file, do_lower_case=True)

    tpu_cluster_resolver = None
    if hparams.use_tpu and hparams.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            hparams.tpu_name, zone=hparams.tpu_zone,
            project=hparams.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=parsed_args.output_dir,
        save_checkpoints_steps=hparams.save_checkpoint_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=hparams.iterations_per_loop,
            num_shards=hparams.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if hparams.do_train:
        train_examples = processor.get_train_examples(parsed_args.data_dir)
        num_train_steps = int(
            len(
                train_examples) / hparams.batch_size * hparams.epochs)
        num_warmup_steps = int(num_train_steps * hparams.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=hparams.init_checkpoint,
        learning_rate=hparams.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=hparams.use_tpu,
        use_one_hot_embeddings=hparams.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=_EVAL_BATCH_SIZE,
        predict_batch_size=_EVAL_BATCH_SIZE)

    if hparams.do_train:
        train_file = os.path.join(parsed_args.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, hparams.max_seq_length, tokenizer,
            train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", hparams.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = _file_based_input_fn_builder(
            input_file=train_file,
            seq_length=hparams.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if hparams.do_eval:
        eval_examples = processor.get_dev_examples(parsed_args.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if hparams.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the
            # number of examples must be a multiple of the batch size, or else
            # examples will get dropped. So we pad with fake examples which are
            # ignored later on. These do NOT count towards the metric (all
            # tf.metrics support a per-instance weight, and these get a weight
            # of 0.0).
            while len(eval_examples) % _EVAL_BATCH_SIZE != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(parsed_args.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, hparams.max_seq_length, tokenizer,
            eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", _EVAL_BATCH_SIZE)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if hparams.use_tpu:
            assert len(eval_examples) % _EVAL_BATCH_SIZE == 0
            eval_steps = int(len(eval_examples) // _EVAL_BATCH_SIZE)

        eval_drop_remainder = True if hparams.use_tpu else False
        eval_input_fn = _file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=hparams.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(parsed_args.output_dir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    tf.app.run()
