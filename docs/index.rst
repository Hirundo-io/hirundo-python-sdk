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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom unlearning, ``biased_dataset`` is the forget dataset,
``unbiased_dataset`` is the retain dataset, and ``target_utilities`` contains
additional utility datasets. Each ``HuggingFaceDataset`` accepts either an inline
``token`` or a saved ``token_id``. Supplying both raises a local validation error,
including when constructing behaviors or utilities from nested dictionaries.
Omit both for public datasets that allow unauthenticated downloads.

The following example reads one optional ``HF_TOKEN`` from the local environment
and explicitly passes it to source-model registration and every user-configured
Hugging Face dataset. Its editable ``FORGET_DATASETS`` and ``RETAIN_DATASETS``
lists start with the Security preset sources; ``ADDITIONAL_UTILITY_DATASETS``
starts empty because the platform already adds the built-in utility sources.
Replace or extend these lists with compatible sources before running it; calling
``launch`` starts a training job.

.. literalinclude:: llm_custom_dataset_example.py
   :language: python

Set any of ``model_token``, ``forget_token``, ``retain_token``, or ``utility_token``
to a different credential obtained securely by your application for per-resource
overrides, or to ``None`` to leave that resource unauthenticated. Local environment
variables alone do not authenticate server-side downloads: passing a credential
only to model registration does not pass it to datasets. Keep secrets outside
source files and avoid logging credentials or serialized request payloads.

To save an existing Hugging Face credential in Hirundo, create a
``HuggingFaceToken`` with a name unique within your organization. Its ``create``
method returns a numeric ID to reuse on datasets, omitting their inline ``token``.
This registers an existing credential rather than minting a new Hugging Face
access token. The example below reads ``HF_TOKEN``, saves it, registers a model,
and launches custom unlearning with saved credentials for all three datasets.
Replace the resource names before running it. For later runs, set
``HIRUNDO_HF_TOKEN_ID`` to the returned ID so the example skips credential
creation and reuses the saved token instead of creating a duplicate name.

.. literalinclude:: llm_saved_dataset_token_example.py
   :language: python

Omit ``organization_id`` to use your default organization, or pass it when
creating the saved token, registering the model, and constructing ``LlmRunInfo``
to select another organization you can access. The platform checks access to the
run's organization and resolves the saved token within that organization.
Invalid IDs and IDs from other organizations are rejected. The selected credential must
also have access to the dataset, including any required gated-dataset approval.

Custom dataset lists and deployment compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CustomBehavior.biased_dataset`` (forget) and ``unbiased_dataset`` (retain) each
accept either one existing dataset object or a list of 1–32 dataset objects.
The two lists are independent: two forget sources and three retain sources are
valid. Use exactly one ``CustomBehavior``; every source is sent once in its own
role, without pairing, repetition, or cross products. A single-object input
keeps its single-object wire payload. One-item lists remain lists, and either
role can use a single object while the other uses a list.

Each member may be a ``HuggingFaceDataset`` or, where the deployment supports
uploaded CSV sources, a ``HirundoCSVDataset`` using the deployment's upload URL.
Current non-air-gapped deployments reject uploaded CSV sources; list support
does not enable a dataset type that the deployment otherwise rejects.
Equivalent nested dictionaries are accepted and validated, including each
dataset's independent ``token``/``token_id`` choice.
Empty lists, lists exceeding 32 sources per role, invalid members, and conflicting
credentials fail locally. The limit applies to each role independently.

In an air-gapped deployment using local bundled assets, Hugging Face dataset names
resolve to datasets installed in the deployment's asset bundle. Have the operator
stage every configured source and its required files before submitting a run;
credentials do not make an unbundled source available offline. Uploaded CSV
sources must also be staged through that deployment's supported upload mechanism.
Dataset lists do not bypass the deployment's source-access or materialization
checks, and the SDK does not download datasets on the server's behalf.

The platform prepares each source and merges/concatenates the prepared datasets
within its role using the existing core dataset preparation behavior. List
lengths do not define pairs or weights. Existing preprocessing, sampling, and
size limits still apply; this SDK feature does not introduce weighting or
change those server-side settings. ``ADDITIONAL_UTILITY_DATASETS`` adds to the
platform's built-in utilities; it does not replace them. The example deliberately
leaves this list empty to avoid repeating the built-in Security utilities.

**Deployment requirement:** list payloads require matching platform API schema,
persistence, credential handling, and worker support. Before submitting lists,
verify the deployment's ``/openapi.json`` includes the array alternatives with
``minItems: 1`` and ``maxItems: 32`` for both CUSTOM dataset fields, or confirm
support with your deployment operator. The SDK does not probe for this feature.
Older deployments can reject list payloads, including one-item lists.
The SDK propagates the server error without
truncating sources or launching separate training runs. Continue using legacy
single-object inputs until the deployment supports lists. Each source continues
to use its own inline or saved dataset credentials.

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

Google Colab notebooks
----------------------

You can find more examples in the
`Google Colab notebooks <https://github.com/Hirundo-io/hirundo-python-sdk/tree/main/notebooks>`_.
