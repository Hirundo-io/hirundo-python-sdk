.. hirundo documentation master file, created by
   sphinx-quickstart on Sun Jul 21 10:18:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :http-equiv=Content-Security-Policy: default-src 'self'; frame-ancestors 'none'

Hirundo Python SDK
==================

Welcome to the ``hirundo`` client library documentation. This SDK connects to the
Hirundo platform and provides APIs for:

- LLM behavior unlearning runs (reducing bias, prompt injections and other unwanted behaviors).
- LLM behavior eval runs (measuring bias, hallucination, prompt injection, and more).
- Dataset QA for machine learning datasets.

Getting started
---------------

Install the SDK:

.. code-block:: bash

   pip install hirundo

Configure API access:

.. code-block:: bash

   hirundo setup

This writes ``HIRUNDO_API_KEY`` (and optionally ``HIRUNDO_API_HOST``) to a local ``.env`` file or
``~/.hirundo.conf`` for subsequent SDK usage.

LLM behavior unlearning
-----------------------

The SDK can launch unlearning runs to reduce unwanted behaviors in LLMs. A run
targets one or more behaviors (bias, prompt injection, or custom datasets)
and returns an adapter that can be used with Hugging Face Transformers pipelines.

Example:

.. literalinclude:: llm_unlearning_example.py
   :language: python

Custom Hugging Face dataset credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom unlearning, ``biased_dataset`` is the forget dataset,
``unbiased_dataset`` is the retain dataset, and ``target_utilities`` contains
additional utility datasets. Each ``HuggingFaceDataset`` accepts either an inline
``token`` or a saved ``token_id``. Supplying both raises a local validation error,
including when constructing behaviors or utilities from nested dictionaries.
Omit both for public datasets that allow unauthenticated downloads.

The following example reads one optional ``HF_TOKEN`` from the local environment
and explicitly passes it to source-model registration and every user-configured
Hugging Face dataset. Replace the dataset names with your own compatible datasets
before running it; calling ``launch`` starts a training job.

.. literalinclude:: llm_custom_dataset_example.py
   :language: python

Set any of ``model_token``, ``forget_token``, ``retain_token``, or ``utility_token``
to a different credential obtained securely by your application for per-resource
overrides, or to ``None`` to leave that resource unauthenticated. Local environment
variables alone do not authenticate server-side downloads: passing a credential
only to model registration does not pass it to datasets. Keep secrets outside
source files and avoid logging credentials or serialized request payloads.

To reference a Hugging Face token already saved in the Hirundo platform, use its
numeric ID on each dataset that should use it, omitting ``token``:

.. code-block:: python

   from hirundo.unlearning_llm import HuggingFaceDataset

   retain_dataset = HuggingFaceDataset(
       hugging_face_dataset_name="your-org/retain-dataset",
       token_id=123,  # Replace with your saved Hirundo Hugging Face token ID.
   )

Use this dataset in ``CustomBehavior`` or ``CustomUtility`` just like the inline
credential examples. The platform checks access to the run's organization and
resolves the saved token within that organization; an invalid or different
organization's token ID is rejected. The selected Hugging Face credential must
also have access to the dataset, including any required gated-dataset approval.

LLM behavior eval
-----------------

Run standardized evaluations over an LLM or an unlearning run to quantify
behavior changes (bias, hallucination, prompt injections, and more).

Example:

.. literalinclude:: llm_behavior_eval_example.py
   :language: python

External evaluations
--------------------

Run server-owned Inspect evaluations. The deployment catalogue determines which
tasks are available, so applications should retrieve it before launching a run.

.. literalinclude:: external_eval_example.py
   :language: python

Dataset QA
----------

You can run QA on datasets without sharing your training code. The API supports
multiple labeling types, including:

- Single-label classification
- Object detection
- Object/semantic/panoptic segmentation
- Speech-to-text
- Tabular classification
- Timeseries classification

Supported storage backends include:

- Amazon S3
- Google Cloud Storage (GCS)
- Git repositories with LFS (GitHub, Hugging Face)
- Local storage for on-premises installations

For on-premises local storage, pass ``StorageTypes.LOCAL`` on the dataset. The
SDK resolves the organization's existing local storage config when creating the
dataset, so examples do not need hard-coded storage config IDs:

.. code-block:: python

   QADataset(
       name="on-prem local dataset",
       labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
       storage_config=StorageTypes.LOCAL,
       labeling_info=HirundoCSV(
           csv_url=Url("file:///datasets/my-dataset/metadata.csv"),
       ),
       modality=ModalityType.TABULAR,
   )

Classification example:

.. literalinclude:: dataset_qa_classification_example.py
   :language: python

Object detection example:

.. literalinclude:: dataset_qa_object_detection_example.py
   :language: python

Timeseries example:

.. literalinclude:: dataset_qa_timeseries_example.py
   :language: python

Multimodal example:

.. literalinclude:: dataset_qa_multimodal_example.py
   :language: python

API reference
-------------

.. toctree::
   :maxdepth: 2

   modules
   release_notes

Google Colab notebooks
----------------------

You can find more examples in the
`Google Colab notebooks <https://github.com/Hirundo-io/hirundo-python-sdk/tree/main/notebooks>`_.
