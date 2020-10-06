"""Train and evaluate an intent classifier based on a sentence encoder

usage:
    python run_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.default \
  --output_dir=<OUTPUT_DIR> \
  --params_overrides task=${DS},data_regime=${DR},encoder_type=${ENC}

Copyright PolyAI Limited.
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from datetime import datetime
import csv
import json

import pandas as pd
import glog
import numpy as np
import tensorflow as tf
import resource
from intent_detection.classifier import train_model
from intent_detection.encoder_clients import get_encoder_client
from intent_detection.utils import parse_args_and_hparams
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

_TRAIN = "train"
_TEST = "test"

def online_learning_sim_class_sample(train, nsamples=1):
  
    if nsamples=='full':
        return train
    else:
        nsamples=int(nsamples)
        
    newdf=pd.DataFrame(columns=['text','labels'])
    for c in train.labels.unique():
        temp=train.loc[train.labels==c][:nsamples]
        newdf = newdf.append(temp)
    #     break
    return newdf.reset_index(drop=True)

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

    if hparams.task=='bank_split':
        train=pd.read_csv('/workspace/polyai-models/banking_data/200911_train60_banking_data.csv')
        val=pd.read_csv('/workspace/polyai-models/banking_data/200911_val10_banking_data.csv')
        test=pd.read_csv('/workspace/polyai-models/banking_data/200911_test30_banking_data.csv')
    elif hparams.task=='alliance':
        train = pd.read_csv('/workspace/polyai-models/alliance_split_human/200908_alliance_train_60_human_paraphrase.csv')
        val = pd.read_csv('/workspace/polyai-models/alliance_split_human/200908_alliance_val_10_human_paraphrase.csv')
        test = pd.read_csv('/workspace/polyai-models/alliance_split_human/200908_alliance_test_30_human_paraphrase.csv')
    else:
        print('SELECT AGAIN.')
    train = train.rename(columns={'question':'text'})
    test = test.rename(columns={'question':'text'})
    val = val.rename(columns={'question':'text'})
    train = train.append(val)
    train.reset_index(drop=True, inplace=True)
    train = train[['text','labels']]
    val = val [['text','labels']]
    test = test[['text','labels']]
    
    if over_ride_no_classes!='full':
        most_freq_classes = train.labels.value_counts().index[:over_ride_no_classes]
        train = train.loc[train.labels.isin(most_freq_classes)]
        test = test.loc[test.labels.isin(most_freq_classes)]
        le=LabelEncoder()
        train['labels'] = le.fit_transform(train.labels)
        test['labels'] = le.transform(test.labels)

    
    train = online_learning_sim_class_sample(train, nsamples=hparams.data_regime)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    display(train.head())
            
    print('TRAIN SHAPE',train.shape)
    categories = train.labels.unique()
    labels[_TRAIN]=train.labels.values
    labels[_TEST]=test.labels.values
    if algo not in ['rf_tfidf','sbert_cosine']:

        encodings[_TRAIN] = encoder_client.encode_sentences(train.text.values)
        
        encodings[_TEST] = encoder_client.encode_sentences(test.text.values)
#         if hparams.data_regime == "full":
#             train_file = "train"
#         elif hparams.data_regime == "10":
#             train_file = "train_10"
#         elif hparams.data_regime == "30":
#             train_file = "train_30"
#         else:
#             glog.error(f"Invalid data regime: {hparams.data_regime}")
#         train_data = os.path.join(
#             data_dir, hparams.task, f"{train_file}.csv")
#         test_data = os.path.join(data_dir, hparams.task, "test.csv")
#         categories_file = os.path.join(data_dir, hparams.task, "categories.json")

#         with tf.gfile.Open(categories_file, "r") as categories_file:
#             categories = json.load(categories_file)



#         with tf.gfile.Open(train_data, "r") as data_file:
#             data = np.array(list(csv.reader(data_file))[1:])
#             labels[_TRAIN] = data[:, 1]
#             encodings[_TRAIN] = encoder_client.encode_sentences(data[:, 0])

#         with tf.gfile.Open(test_data, "r") as data_file:
#             data = np.array(list(csv.reader(data_file))[1:])
#             labels[_TEST] = data[:, 1]
#             encodings[_TEST] = encoder_client.encode_sentences(data[:, 0])

#         # convert labels to integers
#         labels = {
#             k: np.array(
#                 [categories.index(x) for x in v]) for k, v in labels.items()
#         }

    return categories, encodings, labels, train, test

def rf_tfidf(train, test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    model = RandomForestClassifier()
    trans = TfidfVectorizer()
    train_text = trans.fit_transform(train.text)
    test_text = trans.transform(test.text)
    train['labels']=train.labels.astype(int)
    print(train.labels.dtype)
    model.fit(train_text, train.labels.values)
    pred = model.predict(test_text)
    acc = accuracy_score(test.labels, pred)
    return acc

def sbert_cosine(train, test):    
    from sentence_transformers import SentenceTransformer
    import time
    import scipy

    # sbert_model = 'roberta-large-nli-stsb-mean-tokens' #'distilbert-base-nli-mean-tokens'
    sbert_model = 'distiluse-base-multilingual-cased'

    model = SentenceTransformer(sbert_model)

#     train_ques_embd = [model.encode(t)[0] for t in train.text]

    print("SBERT + COSINE Model")
    test_ques=test.text
    # res1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start_time = time.time()
    # train_ques_embd = [model.encode(t)[0] for t in train.text]
    train_ques_embd=model.encode(train.text)
    end_time = time.time()
    test_ques = model.encode(test.text)

    train_intn=train.labels
    test_intn=test.labels

    # res2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # print("Train Time = " + str(end_time-start_time) + " sec.")
    # print("Memory Usage = " + str(res2-res1+res) + " MB")

    top = 5
    count = 0
    count1 = 0
    time_count = []
    for c, i in enumerate(test_ques):
        #     start_time = time.time()
        #     test_q = test_ques[i]
        #     test_q_emb = model.encode(test_q)
            dist_vec = scipy.spatial.distance.cdist([i], train_ques_embd, 'cosine')[0]
            pred_intn = train_intn[np.argmin(dist_vec)]
#             pred_top_indx = np.argpartition(dist_vec, top)[:top]
#             pred_top_intn = [train_intn[indx] for indx in pred_top_indx]
        #     end_time = time.time()
        #     time_count.append(end_time-start_time)
            #
            if test_intn[c] == pred_intn:
                count = count + 1
#             if test_intn[c] in pred_top_intn:
#                 count1 = count1 + 1


#     avg_inf_time = sum(time_count) / len(test_ques) * 1000
    p1 = count / len(test_ques)
    print("P@1 = " + str(count / len(test_ques)))
#     print("P@5 = " + str(count1 / len(test_ques)))
#     print("Inference Time = " + str(avg_inf_time) + " msec.")
    return p1

def _main():
    parsed_args, hparams = parse_args_and_hparams()
    
    hparams.data_regime=over_ride_sample_no

    if hparams.task.lower() not in ["clinc", "hwu", "banking", "wallet", "alliance", "bank_split"]:
        raise ValueError(f"{hparams.task} is not a valid task")
    hparams.task = over_ride_dataset
    hparams.encoder_type = algo
    
    encoder_client = get_encoder_client(hparams.encoder_type,
                                        cache_dir=hparams.cache_dir)

    categories, encodings, labels, train, test = _preprocess_data(
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

        if algo=='rf_tfidf':
            acc = rf_tfidf(train, test)
            return over_ride_sample_no, acc
        elif algo == 'sbert_cosine':
            acc = sbert_cosine(train, test)
            return over_ride_sample_no, acc
        else:
            model, eval_acc_history = train_model(
            encodings[_TRAIN], labels[_TRAIN], categories, hparams,
            validation_data=validation_data, verbose=verbose)
            _, acc = model.evaluate(encodings[_TEST], labels[_TEST], verbose=0)
#         print(_, 'loss of evaluation')
#         print('PREDICT')
#         pred = model.predict(encodings[_TEST], labels[_TEST])
        
#         t = []
#         for r in [5]:
#             topk = pred.argsort(axis=1)[:,-r:]#[::-1]
#             t.append(sum([1 if i in topk[c] else 0 for c,i in enumerate(labels[_TEST])])/labels[_TEST].shape[0])
#         print('p@1',acc)
# #         print('p@5',t[0])
#         print('memory:',memory_pred)
        
        glog.info(f"Seed accuracy: {acc:.3f}")
        accs.append(acc)
        eval_acc_histories.append(eval_acc_history)

    average_acc = np.mean(accs)
    variance = np.std(accs)
    glog.info(
        f"Average results:\n"
        f"Accuracy {over_ride_sample_no}: {average_acc:.3f}\n"
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
        
    memory_pred = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print('MEMORY_PRED:', memory_pred)
    
    
    start_time = datetime.now()
    query_encoding = encoder_client.encode_sentences(['test'])
    output = model.predict(query_encoding)
    prediction = np.argmax(output)
    end_time = datetime.now()
    delta = end_time - start_time
    print('TIME DELTA:', delta)
    return (over_ride_sample_no, acc, memory_pred, delta, hparams.task)

if __name__ == "__main__":
    for algo in ['rf_tfidf']: #['sbert','use','convert','combined','laser_convert_use']:'sbert_cosine',
        for over_ride_dataset in [ "wallet", "alliance", "bank_split"]:
            res = pd.DataFrame(columns=[3,5,10,15,20,100, 'full'], index=[1,2,3,5,10,20, 'full'])
            for over_ride_sample_no in res.index:
                for over_ride_no_classes in res.columns:
                    mainout=_main()
                    res.loc[over_ride_sample_no,over_ride_no_classes]=mainout[1]
            res.to_csv(f'200930_{algo}_{over_ride_dataset}.csv')
            print(over_ride_dataset, algo)
            print(res)

