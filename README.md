# Dockerfile added....

After conflicts with tf_sentencepiece and tensorflow_text within conda env pointed to the necessity of a dockerfile. 

Other useful commands:
```python
docker build -t tensorflow114 .
```
```
docker run -it --gpus all -p 9000:9000 -v /media/aiops/nvme_new/EDWARD/nlp_code/tf_docker_share:/workspace/polyai-models tensorflow114
```
Run a notebook and expose port 9000 to the internet: 
```
nohup jupyter notebook --ip 0.0.0.0 --port 9000 --no-browser --allow-root &
```
cat nohup, to get the token...  

<br/><br/>

[![PolyAI](polyai-logo.png)](https://poly-ai.com/)

[![CircleCI](https://circleci.com/gh/PolyAI-LDN/polyai-models.svg?style=svg&circle-token=51b384ab1be46e42b3f007fa2d9cfdb31b7599e4)](https://circleci.com/gh/PolyAI-LDN/polyai-models)

# polyai-models

*Neural Models for Conversational AI*

This repo shares models from [PolyAI](https://poly-ai.com) publications, including the *ConveRT* efficient dual-encoder model. These are shared as Tensorflow Hub modules, listed below.
We also share example code and utility classes, though for many the
Tensorflow Hub URLs will be enough.


* [Requirements](#requirements)
* [Models](#models)
  * [ConveRT](#convert)
  * [Multi-Context ConveRT](#multi-context-convert)
  * [ConveRT finetuned on Ubuntu](#convert-finetuned-on-ubuntu)
  * [Intent Detection Benchmarks](#intent-detection-benchmarks)
* [Keras layers](#keras-layers)
* [Encoder client](#encoder-client)
* [Citations](#citations)
* [Development](#development)


# Requirements

Using these models requires [Tensorflow Hub](https://www.tensorflow.org/hub) and [Tensorflow Text](https://www.tensorflow.org/tutorials/tensorflow_text/intro). In particular, Tensorflow Text provides ops that allow the model to directly work on text, requiring no pre-processing or tokenization from the user. You must `import tensorflow_text` before loading the tensorflow hub modules, or you will see an error about 'ops missing from the python registry'. The models are compatible with any of the following combinations:

* Tensorflow 1.14 and Tensorflow Text 0.6.0 (*used for tests and examples this repo*)
* Tensorflow 1.15 and Tensorflow Text 1.15.x
* Tensorflow 2.0 and Tensorflow Text 2.0.x

A list of available versions can be found [on the Tensorflow Text github repo](https://github.com/tensorflow/text/releases).
Note for Tensorflow 2.0 you may need to disable eager execution with `tf.compat.v1.disable_eager_execution()`.


# Models

## ConveRT

This is the ConveRT dual-encoder model, using subword representations and lighter-weight more efficient transformer-style
blocks to encode text, as described in [the ConveRT paper](https://arxiv.org/abs/1911.03688).
It provides powerful representations for conversational data, and can also be used as a response ranker.
The model costs under $100 to train from scratch, can be quantized to under 60MB, and is competitive with larger Transformer networks on conversational tasks.
We share an unquantized version of the model, facilitating fine-tuning. Please [get in touch](https://www.polyai.com/contact/) if you are interested in using the quantized ConveRT model. The Tensorflow Hub url is:

```python
module = tfhub.Module("http://models.poly-ai.com/convert/v1/model.tar.gz")
```

See the [`convert-examples.ipynb` notebook](examples/convert-examples.ipynb) for some examples of how to use this model.

### TFHub signatures

 **default**

 Takes as input `sentences`, a string tensor of sentences to encode. Outputs 1024 dimensional vectors, giving a representation for each sentence. These are the output of the sqrt-N reduction in the shared tranformer encoder. These representations work well as input to classification models. Note that these vectors are not normalized in any way, so you might find l2 normalizing them helps for learning, especially when using SGD.

```python
sentence_encodings = module(
  ["hello how are you?", "what is your name?", "thank you good bye"])
```

**encode_context**

 Takes as input `contexts`, a string tensor of contexts to encode. Outputs 512 dimensional vectors, giving the context representation of each input. These are trained to have a high cosine-similarity with the response representations of good responses (from the `encode_response` signature)

```python
context_encodings = module(
  ["hello how are you?", "what is your name?", "thank you good bye"],
  signature="encode_context")
```

**encode_response**

 Takes as input `responses`, a string tensor of responses to encode. Outputs 512 dimensional vectors, giving the response representation of each input. These are trained to have a high cosine-similarity with the context representations of good corresponding contexts (from the `encode_context` signature)

```python
response_encodings = module(
  ["i am well", "I am Matt", "bye!"],
  signature="encode_response")
```

**encode_sequence**

 Takes as input `sentences`, a string tensor of sentences to encode. This outputs sequence encodings, a 3-tensor of shape `[batch_size, max_sequence_length, 512]`, as well as the corresponding subword tokens, a utf8-encoded matrix of shape `[batch_size, max_sequence_length]`. The tokens matrix is padded with empty strings, which may help in masking the sequence tensor. The [`encoder_utils.py`](encoder_utils.py) library has a few functions for dealing with these tokenizations, including a detokenization function, and a function that infers byte spans in the original strings.  


```python
output = module(
  ["i am well", "I am Matt", "bye!"],
  signature="encode_sequence", as_dict=True)
sequence_encodings = output['sequence_encoding']
tokens = output['tokens']
```

**tokenize**

Takes as input `sentences`, a string tensor of sentences to encode. This outputs the corresponding subword tokens, a utf8-encoded matrix of shape `[batch_size, max_sequence_length]`. The tokens matrix is padded with empty strings. Usually this process is internal to the network, but for some applications it may be useful to access the internal tokenization.

```python
tokens = module(
  ["i am well", "I am Matt", "bye!"],
  signature="tokenize")
```

## Multi-Context ConveRT

This is the multi-context ConveRT model from [the ConveRT paper](https://arxiv.org/abs/1911.03688), that uses extra contexts from the conversational history to refine the context representations. This is an unquantized version of the model. The Tensorflow Hub url is:

```python
module = tfhub.Module("http://models.poly-ai.com/multi_context_convert/v1/model.tar.gz")
```

### TFHub signatures

This model has the same signatures as the ConveRT encoder, except for the `encode_context` signature that also takes the extra contexts as input. The extra contexts are the previous messages in the dialogue (typically at most 10) prior to the immediate context, and must be joined with spaces from most recent to oldest.

For example, consider the dialogue:

A: Hey!
B: Hello how are you?
A: Fine, strange weather recently right?
B: Yeah

then the context representation is computed as:

```python
context = ["Yeah"]
extra_context = ["Fine, strange weather recently right? Hello how are you? Hey!"]
context_encodings = module(
  {
    'context': context,
    'extra_context': extra_context,
  },
  signature="encode_context",
)
```

See [`encoder_client.py`](encoder_client.py) for code that computes these features.


## ConveRT finetuned on Ubuntu

This is the multi-context ConveRT model, fine-tuned to the DSTC7 Ubuntu response ranking task. It has the exact same signatures as the extra context model, and has TFHub uri `http://models.poly-ai.com/ubuntu_convert/v1/model.tar.gz`. Note that this model requires prefixing the extra context features with `"0: "`, `"1: "`, `"2: "` etc.

The [`dstc7/evaluate_encoder.py`](dstc7/evaluate_encoder.py) script demonstrates using this encoder to reproduce the results from [the ConveRT paper](https://arxiv.org/abs/1911.03688).

## Intent Detection Benchmarks

A set of intent detectors trained on top of ConveRT and other sentence encoders can be found in the [`intent_detection`](intent_detection) directory. These are the intent detectors presented in [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/pdf/1903.05566.pdf). 

# Keras layers

Keras layers for the above encoder models are implemented in [`encoder_layers.py`](encoder_layers.py). These may be useful for building a model that extends the encoder models, and/or fine-tuning them to your own data.

# Encoder client

A python class `EncoderClient` is implemented in [`encoder_client.py`](encoder_client.py), that gives a simple interface for encoding sentences, contexts, and responses with the above models. It takes python strings as input, and numpy matrices as output:

```python
client = encoder_client.EncoderClient(
    "http://models.poly-ai.com/convert/v1/model.tar.gz")

# We will find good responses to the following context.    
context_encodings = client.encode_contexts(["What's your name?"])

# Let's rank the following responses as candidates.
candidate_responses = ["No thanks.", "I'm Matt.", "Hey.", "I have a dog."]
response_encodings = client.encode_responses(candidate_responses)

# The scores are computed using the dot product.
scores = response_encodings.dot(context_encodings.T).flatten()

# Output the top scoring response.
top_idx = scores.argmax()
print(f"Best response: {candidate_responses[top_idx]}, score: {scores[top_idx]:.3f}")

# This should print "Best response: I'm Matt., score: 0.377".
```

Internally it implements caching, deduplication, and batching, to help speed up encoding. Note that because it does batching internally, you can pass very large lists of sentences to encode without going out of memory.

# Citations

* [ConveRT: Efficient and Accurate Conversational Representations from Transformers](https://arxiv.org/abs/1911.03688)
```bibtext
@article{Henderson2019convert,
    title={{ConveRT}: Efficient and Accurate Conversational Representations from Transformers},
    author={Matthew Henderson and I{\~{n}}igo Casanueva and Nikola Mrk\v{s}i\'{c} and Pei-Hao Su and Tsung-Hsien and Ivan Vuli\'{c}},
    journal={CoRR},
    volume={abs/1911.03688},
    year={2019},
    url={http://arxiv.org/abs/1911.03688},
}
```

# Development

Setting up an environment for development:

* Create a python 3 virtual environment

```bash
python3 -m venv ./venv
```

* Install the requirements

```bash
. venv/bin/activate
pip install -r requirements.txt
```

* Run the unit tests

```bash
python -m unittest discover -p '*_test.py' .
```

Pull requests will trigger a CircleCI build that:

* runs flake8 and isort
* runs the unit tests
