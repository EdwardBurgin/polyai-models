"""Config files for sentence encoder and finetuned bert based classifiers

Copyright PolyAI Limited.
"""

from copy import deepcopy

import tensorflow as tf

default = tf.contrib.training.HParams(
    # model hparams
    epochs=500,
    learning_rate=0.7,
    lr_decay_pow=1,
    batch_size=32,
    num_hidden_layers=1,
    hidden_layer_size=512,
    activation="relu",
    dropout=0.75,
    optimizer="sgd",
    encoder_type="convert",

    # training hparams
    cache_dir="",
    balance_data=False,
    task="banking",
    data_regime="full",
    eval_each_epoch=False,
    seeds=10
)


pretrained_bert_dir = "<PRETRAINED_BERT_DIR>"
# Note: You will have to download the pretrained bert model from
# https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip  # NOQA
# and point pretrained_bert_dir to the download path

bert_ft = tf.contrib.training.HParams(
    # pretrained bert files
    bert_config_file=f"{pretrained_bert_dir}/bert_config.json",
    vocab_file=f"{pretrained_bert_dir}/vocab.txt",
    init_checkpoint=f"{pretrained_bert_dir}/bert_model.ckpt",

    # model params
    batch_size=32,
    learning_rate=4e-5,
    epochs=5,
    warmup_proportion=0.1,

    # training params
    task="banking",
    data_regime="full",
    max_seq_length=90,
    do_train=True,
    do_eval=True,
    save_checkpoint_steps=1000,
    iterations_per_loop=1000,

    # tpu params
    use_tpu=False,
    tpu_name=None,
    tpu_zone=None,
    gcp_project=None,
    num_tpu_cores=8
)

bert_ft_tpu = deepcopy(bert_ft)
bert_ft_tpu.use_tpu = True
# delete hparams to reset the expected type, otherwise it will conflict with
# --params_override
bert_ft_tpu.del_hparam("tpu_name")
bert_ft_tpu.add_hparam("tpu_name", "v2-8")
bert_ft_tpu.del_hparam("tpu_zone")
bert_ft_tpu.add_hparam("tpu_zone", "europe-west4-a")
bert_ft_tpu.del_hparam("gcp_project")
bert_ft_tpu.add_hparam("gcp_project", "<GCP_PROJECT>")

bert_ft_tpu_banking = deepcopy(bert_ft_tpu)

bert_ft_tpu_banking_10 = deepcopy(bert_ft_tpu_banking)
bert_ft_tpu_banking_10.data_regime = "10"
bert_ft_tpu_banking_10.epochs = 50

bert_ft_tpu_banking_30 = deepcopy(bert_ft_tpu_banking)
bert_ft_tpu_banking_30.data_regime = "30"
bert_ft_tpu_banking_30.epochs = 17

bert_ft_tpu_clinc = deepcopy(bert_ft_tpu)
bert_ft_tpu_clinc.max_seq_length = 33
bert_ft_tpu_clinc.task = "clinc"

bert_ft_tpu_clinc_10 = deepcopy(bert_ft_tpu_clinc)
bert_ft_tpu_clinc_10.data_regime = "10"
bert_ft_tpu_clinc_10.epochs = 50

bert_ft_tpu_clinc_30 = deepcopy(bert_ft_tpu_clinc)
bert_ft_tpu_clinc_30.data_regime = "30"
bert_ft_tpu_clinc_30.epochs = 17

bert_ft_tpu_hwu = deepcopy(bert_ft_tpu)
bert_ft_tpu_hwu.max_seq_length = 30
bert_ft_tpu_hwu.task = "hwu"

bert_ft_tpu_hwu_10 = deepcopy(bert_ft_tpu_hwu)
bert_ft_tpu_hwu_10.data_regime = "10"
bert_ft_tpu_hwu_10.epochs = 50

bert_ft_tpu_hwu_30 = deepcopy(bert_ft_tpu_hwu)
bert_ft_tpu_hwu_30.data_regime = "30"
bert_ft_tpu_hwu_30.epochs = 17
