"""Sentence encoder library for tensorflow_hub based sentence encoders

The included sentence encoders are:
- BERT: https://arxiv.org/abs/1810.04805
- USE multilingual: https://arxiv.org/abs/1907.04307
- ConveRT: https://arxiv.org/abs/1911.03688

Copyright PolyAI Limited.
"""

import abc
import os
import pickle

import glog
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text  # NOQA: it is used when importing ConveRT.
import tf_sentencepiece  # NOQA: it is used when importing USE.
from bert.tokenization import FullTokenizer
from tqdm import tqdm

from encoder_client import EncoderClient

from laserembeddings import Laser

from sentence_transformers import SentenceTransformer
sbert_model = 'roberta-large-nli-stsb-mean-tokens' #'distilbert-base-nli-mean-tokens'



_CONVERT_PATH = "http://models.poly-ai.com/convert/v1/model.tar.gz"
_USE_PATH = ("https://tfhub.dev/google/universal-sentence-encoder-"
             "multilingual-large/1")
_BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"


def l2_normalize(encodings):
    """L2 normalizes the given matrix of encodings."""
    norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
    return encodings / norms


class ClassificationEncoderClient(object):
    """A model that maps from text to dense vectors."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode_sentences(self, sentences):
        """Encodes a list of sentences

        Args:
            sentences: a list of strings

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        return NotImplementedError


def get_encoder_client(encoder_type, cache_dir=None):
    """get an EncoderClient object

    Args:
        encoder_type: (str) one of "use", "convert", "combined", "laser_convert" or "bert"
        cache_dir: The directory where an encoding dictionary will be cached

    Returns:
        a ClassificationEncoderClient

    """
    if encoder_type.lower() == "use":
        encoder_client = UseEncoderClient(_USE_PATH)
        if cache_dir:
            encoder_id = _USE_PATH.replace("/", "-")
            encoder_client = CachingEncoderClient(
                encoder_client, encoder_id, cache_dir)
    elif encoder_type.lower() == "convert":
        encoder_client = ConvertEncoderClient(_CONVERT_PATH)
        if cache_dir:
            encoder_id = _CONVERT_PATH.replace("/", "-")
            encoder_client = CachingEncoderClient(
                encoder_client, encoder_id, cache_dir)
    elif encoder_type.lower() == "combined":
        use_encoder = UseEncoderClient(_USE_PATH)
        convert_encoder = ConvertEncoderClient(_CONVERT_PATH)
        if cache_dir:
            use_id = _USE_PATH.replace("/", "-")
            use_encoder = CachingEncoderClient(
                use_encoder, use_id, cache_dir)
            convert_id = _CONVERT_PATH.replace("/", "-")
            convert_encoder = CachingEncoderClient(
                convert_encoder, convert_id, cache_dir)
        encoder_client = CombinedEncoderClient([convert_encoder, use_encoder])
    elif encoder_type.lower() == "bert":
        encoder_client = BertEncoderClient(_BERT_PATH)
        if cache_dir:
            encoder_id = _BERT_PATH.replace("/", "-")
            encoder_client = CachingEncoderClient(
                encoder_client, encoder_id, cache_dir)
    elif encoder_type.lower() == "laser_convert":
        laser_encoder = LaserEncoderClient()
        convert_encoder = ConvertEncoderClient(_CONVERT_PATH)
        encoder_client = CombinedEncoderClient([convert_encoder, laser_encoder])
    elif encoder_type.lower() == "laser_convert_use":
        use_encoder = UseEncoderClient(_USE_PATH)
        laser_encoder = LaserEncoderClient()
        convert_encoder = ConvertEncoderClient(_CONVERT_PATH)
        encoder_client = CombinedEncoderClient([convert_encoder, laser_encoder, use_encoder])
    elif encoder_type.lower() == "sbert":
        encoder_client = SbertEncoderClient()
    
    else:
        raise ValueError(f"{encoder_type} is not a valid encoder type")
    return encoder_client


class CachingEncoderClient(ClassificationEncoderClient):
    """Wrapper around an encoder to cache the encodings on disk"""
    def __init__(self, encoder_client, encoder_id, cache_dir):
        """Create a new CachingEncoderClient object

        Args:
            encoder_client: An EncoderClient
            encoder_id: An unique ID for the encoder
            cache_dir: The directory where the encodings will be cached
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._encodings_dict_path = os.path.join(
            cache_dir, encoder_id)
        self._encoder_client = encoder_client
        self._encodings_dict = self._load_or_create_encodings_dict()

    def _load_or_create_encodings_dict(self):
        if os.path.exists(self._encodings_dict_path):
            with open(self._encodings_dict_path, "rb") as f:
                encodings_dict = pickle.load(f)
        else:
            encodings_dict = {}
        return encodings_dict

    def _save_encodings_dict(self):
        with open(self._encodings_dict_path, "wb") as f:
            pickle.dump(self._encodings_dict, f)

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        missing_sentences = [
            sentence for sentence in sentences
            if sentence not in self._encodings_dict]
        if len(sentences) != len(missing_sentences):
            glog.info(f"{len(sentences) - len(missing_sentences)} cached "
                      f"sentences will not be encoded")
        if missing_sentences:
            missing_encodings = self._encoder_client.encode_sentences(
                missing_sentences)
            for sentence, encoding in zip(missing_sentences,
                                          missing_encodings):
                self._encodings_dict[sentence] = encoding
            self._save_encodings_dict()

        encodings = np.array(
            [self._encodings_dict[sentence] for sentence in sentences])
        return encodings

class LaserEncoderClient(ClassificationEncoderClient):
    """A wrapper around ClassificationEncoderClient to normalise the output"""
    def __init__(self, batch_size=100):
        """Create a new ConvertEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._encoder_client = Laser()         

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        glog.setLevel("ERROR")
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._encoder_client.embed_sentences(
                    sentences[i:i + self._batch_size], lang='en'))
        glog.setLevel("INFO")
        print('DEBUG LASER SIZE:',np.vstack(encodings).shape)
        return l2_normalize(np.vstack(encodings))
    
class SbertEncoderClient(ClassificationEncoderClient):
    """A wrapper around ClassificationEncoderClient to normalise the output"""
    def __init__(self, batch_size=100):
        """Create a new ConvertEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._encoder_client = SentenceTransformer(sbert_model)        

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        glog.setLevel("ERROR")
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._encoder_client.encode(
                    sentences[i:i + self._batch_size]))
        glog.setLevel("INFO")
        print('DEBUG SBERT SIZE:',np.vstack(encodings).shape)
        return l2_normalize(np.vstack(encodings))  
    
class ConvertEncoderClient(ClassificationEncoderClient):
    """A wrapper around ClassificationEncoderClient to normalise the output"""
    def __init__(self, uri, batch_size=100):
        """Create a new ConvertEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._encoder_client = EncoderClient(uri)

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        glog.setLevel("ERROR")
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._encoder_client.encode_sentences(
                    sentences[i:i + self._batch_size]))
        glog.setLevel("INFO")
        print('DEBUG CONVERT SIZE:', np.vstack(encodings).shape)
        return l2_normalize(np.vstack(encodings))


class UseEncoderClient(ClassificationEncoderClient):
    """A Universal Sentence Encoder model loaded as a tensorflow hub module"""
    def __init__(self, uri, batch_size=100):
        """Create a new UseEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub USE module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tf_hub.Module(uri)
            self._fed_texts = tf.placeholder(shape=[None], dtype=tf.string)
            self._embeddings = embed_fn(self._fed_texts)
            encoding_info = embed_fn.get_output_info_dict().get('default')
            if encoding_info:
                self._encoding_dim = encoding_info.get_shape()[-1].value
            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._session.run(
                    self._embeddings,
                    {self._fed_texts: sentences[i:i + self._batch_size]}))
        print('DEBUG USE SIZE:', np.vstack(encodings).shape)
        return np.vstack(encodings)


class BertEncoderClient(ClassificationEncoderClient):
    """The BERT encoder that is loaded as a module from tensorflow hub.

    This class tokenizes the input text using the bert tokenization
    library. The final encoding is computed as the sum of the token
    embeddings.

    Args:
        uri: (string) the tensorflow hub URI for the model.
        batch_size: maximum number of sentences to encode at once
    """
    def __init__(self, uri, batch_size=100):
        """Create a new `BERTEncoder` object."""
        if not tf.test.is_gpu_available():
            glog.warning(
                "No GPU detected, BERT will run a lot slower than with a GPU.")

        self._batch_size = batch_size
        self._session = tf.Session(graph=tf.Graph())
        with self._session.graph.as_default():
            glog.info("Loading %s model from tensorflow hub", uri)
            embed_fn = tf_hub.Module(uri, trainable=False)
            self._tokenizer = self._create_tokenizer_from_hub_module(uri)
            self._input_ids = tf.placeholder(
                name="input_ids", shape=[None, None], dtype=tf.int32)
            self._input_mask = tf.placeholder(
                name="input_mask", shape=[None, None], dtype=tf.int32)
            self._segment_ids = tf.zeros_like(self._input_ids)
            bert_inputs = dict(
                input_ids=self._input_ids,
                input_mask=self._input_mask,
                segment_ids=self._segment_ids
            )

            embeddings = embed_fn(
                inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]
            mask = tf.expand_dims(
                tf.cast(self._input_mask, dtype=tf.float32), -1)
            self._embeddings = tf.reduce_sum(mask * embeddings, axis=1)

            init_ops = (
                tf.global_variables_initializer(), tf.tables_initializer())
        glog.info("Initializing graph.")
        self._session.run(init_ops)

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an array with shape (len(sentences), ENCODING_SIZE)
        """
        encodings = []
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._session.run(
                    self._embeddings,
                    self._feed_dict(sentences[i:i + self._batch_size])))
        return l2_normalize(np.vstack(encodings))

    @staticmethod
    def _create_tokenizer_from_hub_module(uri):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = tf_hub.Module(uri, trainable=False)
            tokenization_info = bert_module(
                signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run(
                    [
                        tokenization_info["vocab_file"],
                        tokenization_info["do_lower_case"]
                    ])

        return FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def _feed_dict(self, texts, max_seq_len=128):
        """Create a feed dict for feeding the texts as input.
        This uses dynamic padding so that the maximum sequence length is the
        smaller of `max_seq_len` and the longest sequence actually found in the
        batch. (The code in `bert.run_classifier` always pads up to the maximum
        even if the examples in the batch are all shorter.)
        """
        all_ids = []
        for text in texts:
            tokens = ["[CLS]"] + self._tokenizer.tokenize(text)

            # Possibly truncate the tokens.
            tokens = tokens[:(max_seq_len - 1)]
            tokens.append("[SEP]")
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            all_ids.append(ids)

        max_seq_len = max(map(len, all_ids))

        input_ids = []
        input_mask = []
        for ids in all_ids:
            mask = [1] * len(ids)

            # Zero-pad up to the sequence length.
            while len(ids) < max_seq_len:
                ids.append(0)
                mask.append(0)

            input_ids.append(ids)
            input_mask.append(mask)

        return {self._input_ids: input_ids, self._input_mask: input_mask}


class CombinedEncoderClient(ClassificationEncoderClient):
    """concatenates the encodings of several ClassificationEncoderClients

    Args:
        encoders: A list of ClassificationEncoderClients
    """
    def __init__(self, encoders: list):
        """constructor"""
        self._encoders = encoders

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an array with shape (len(sentences), ENCODING_SIZE)
        """
        encodings = np.hstack([encoder.encode_sentences(sentences)
                               for encoder in self._encoders])
        print('DEBUG combined size:',encodings.shape)
        return encodings
