.. hirundo documentation master file, created by
   sphinx-quickstart on Sun Jul 21 10:18:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :http-equiv=Content-Security-Policy: default-src 'self', frame-ancestors 'none'

Hirundo Python SDK
==================

Welcome to the ``hirundo`` client library documentation. This SDK connects to the
Hirundo platform and provides APIs for:

- LLM behavior unlearning runs (bias, hallucination, and safety behaviors).
- Dataset QA for machine learning datasets.

Getting started
---------------

Install the SDK:

.. code-block:: bash

   pip install hirundo

Configure API access:

.. code-block:: bash

   hirundo setup

This writes ``API_KEY`` (and optionally ``API_HOST``) to a local ``.env`` file or
``~/.hirundo.conf`` for subsequent SDK usage.

LLM behavior unlearning
-----------------------

The SDK can launch unlearning runs to reduce unwanted behaviors in LLMs. A run
targets one or more behaviors (bias, hallucination, security, or custom datasets)
and returns an adapter that can be used with Hugging Face Transformers pipelines.

Example:

.. literalinclude:: _examples.py
   :language: python
   :start-after: llm-unlearning-example-start
   :end-before: llm-unlearning-example-end

Dataset QA
----------

You can run QA on datasets without sharing your training code. The API supports
multiple labeling types, including:

- Single-label classification
- Object detection
- Object/semantic/panoptic segmentation
- Speech-to-text
- Tabular classification

Supported storage backends include:

- Amazon S3
- Google Cloud Storage (GCS)
- Git repositories with LFS (GitHub, Hugging Face)

Example:

.. literalinclude:: _examples.py
   :language: python
   :start-after: dataset-qa-example-start
   :end-before: dataset-qa-example-end

API reference
-------------

.. toctree::
   :maxdepth: 2

   modules

Google Colab notebooks
----------------------

You can find more examples in the
`Google Colab notebooks <https://github.com/Hirundo-io/hirundo-python-sdk/tree/main/notebooks>`_.
