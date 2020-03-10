"""Tests for encoder_clients.py

Copyright PolyAI Limited.
"""

import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from intent_detection import encoder_clients


class UseEncoderClientTest(unittest.TestCase):
    """Test UseEncoderClient."""

    @patch("tensorflow_hub.Module")
    def test_encode_sentences(self, mock_module_cls):

        class MockFn(object):
            @staticmethod
            def __call__(inputs):
                self.assertIsInstance(inputs, tf.Tensor)
                self.assertEqual(inputs.dtype, tf.string)
                return tf.ones([tf.shape(inputs)[0], 3])

            @staticmethod
            def get_output_info_dict():
                return {}

        mock_module_cls.return_value = MockFn()

        encoder = encoder_clients.UseEncoderClient("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_sentences(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)


class ConveRTEncoderTest(unittest.TestCase):
    """Test ConveRTEncoder."""

    @patch("tensorflow_hub.Module")
    def test_encode_sentences(self, mock_module_cls):

        def mock_fn(inputs, signature=None):
            self.assertIsInstance(inputs, tf.Tensor)
            self.assertEqual(inputs.dtype, tf.string)
            return tf.ones([tf.shape(inputs)[0], 3])

        mock_module_cls.return_value = mock_fn

        encoder = encoder_clients.ConvertEncoderClient("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = encoder.encode_sentences(["hello"])
        # Note than normalised encodings are returned
        expected_encodings = np.asarray([[1., 1., 1.]])
        expected_encodings /= math.sqrt(3.)
        np.testing.assert_allclose(expected_encodings, encodings)


class BERTEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a dummy vocabulary file."""
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "hello", "world",
        ]
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as v_writer:
            v_writer.write("".join([x + "\n" for x in vocab_tokens]))
        cls.vocab_file = v_writer.name

    @classmethod
    def tearDownClass(cls):
        """Delete the dummy vocabulary file."""
        os.unlink(cls.vocab_file)

    @patch("tensorflow_hub.Module")
    def test_encode_sentences(self, mock_module_cls):

        def mock_module(inputs=None, signature=None, as_dict=None):
            self.assertTrue(as_dict)
            if signature == "tokens":
                self.assertEqual(
                    {'input_mask', 'input_ids', 'segment_ids'},
                    inputs.keys())
                batch_size = tf.shape(inputs['input_ids'])[0]
                seq_len = tf.shape(inputs['input_ids'])[1]
                return {
                    'sequence_output': tf.ones([batch_size, seq_len, 3])
                }
            self.assertEqual("tokenization_info", signature)
            return {
                'do_lower_case': tf.constant(True),
                'vocab_file': tf.constant(self.vocab_file),
            }

        mock_module_cls.return_value = mock_module

        encoder = encoder_clients.BertEncoderClient("test_uri")
        self.assertEqual(
            [(("test_uri",), {'trainable': False})] * 2,
            mock_module_cls.call_args_list)

        encodings = encoder.encode_sentences(["hello world"])
        expected_encodings = np.asarray([[1., 1., 1.]])
        expected_encodings /= math.sqrt(3.)
        np.testing.assert_allclose(expected_encodings, encodings)


class CombinedEncoderClientTest(unittest.TestCase):
    def test_encode_sentences(self):
        class Encoder1(encoder_clients.ClassificationEncoderClient):
            def encode_sentences(self, sentences):
                return np.array([[1, 1, 1]] * len(sentences))

        class Encoder2(encoder_clients.ClassificationEncoderClient):
            def encode_sentences(self, sentences):
                return np.array([[2, 2, 2, 2]] * len(sentences))

        combined = encoder_clients.CombinedEncoderClient(
            encoders=[Encoder1(), Encoder2()]
        )
        encodings = combined.encode_sentences(["hello"])
        np.testing.assert_allclose(encodings, [[1, 1, 1, 2, 2, 2, 2]])


class L2NormalizeTest(unittest.TestCase):
    def test_fuzz(self):
        """Test l2_normalize with random inputs."""

        def l2_normalize_python(encodings):
            """Implement l2 normalize in python."""
            out = []
            for encoding in encodings:
                l2_norm = 0.0
                for value in encoding:
                    l2_norm += value * value
                l2_norm = math.sqrt(l2_norm)
                out.append(
                    [value / l2_norm for value in encoding]
                )
            return out

        # Compare numpy l2_normalize with l2_normalize_python.
        for i in range(100):
            # Test with a variety of input shapes.
            num_encodings = (i % 5) + 1
            encoding_dim = (i % 7) + 2
            encodings = np.random.uniform(size=[num_encodings, encoding_dim])
            norm_np = encoder_clients.l2_normalize(encodings)
            self.assertEqual(
                [num_encodings, encoding_dim], list(norm_np.shape))
            norm_py = l2_normalize_python(encodings)
            np.testing.assert_allclose(norm_py, norm_np)


if __name__ == "__main__":
    unittest.main()
