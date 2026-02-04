# Hirundo Python SDK

The Hirundo Python SDK lets you:

- Launch and monitor LLM behavior unlearning runs.
- Run LLM behavior evaluations for bias, hallucination, and prompt injection.
- Run dataset QA for ML datasets (classification, object detection, and more).
- Fetch QA results as `pandas` or `polars` DataFrames.

This SDK requires access to a Hirundo server (SaaS, VPC, or on-prem).

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13 (CPython).
- A Hirundo API key.

## Installation

```bash
pip install hirundo
```

Optional extras:

- LLM behavior unlearning (Transformers + PEFT): `pip install hirundo[transformers]`
- Dataset QA or LLM behavior eval results as DataFrames: `pip install hirundo[pandas]` or `pip install hirundo[polars]`

If you want to install from source, clone this repository and run:

```bash
pip install .
```

## Configure API access

You can set environment variables directly or use the CLI helper:

```bash
hirundo setup
```

This writes `API_KEY` (and optionally `API_HOST`) to `.env` in the current directory or `~/.hirundo.conf`.

## Quickstart: LLM behavior unlearning

Make sure you have the `transformers` extra installed (`pip install hirundo[transformers]`).

```python
from hirundo import (
    BiasRunInfo,
    BiasType,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmUnlearningRun,
)

llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()

run_id = LlmUnlearningRun.launch(
    llm_id,
    BiasRunInfo(bias_type=BiasType.ALL),
)

result = LlmUnlearningRun.check_run(run_id)
new_adapter = llm.get_hf_pipeline_for_run(run_id)
```

## Quickstart: LLM behavior eval

If you want results as DataFrames, install `hirundo[pandas]` or `hirundo[polars]`.

```python
from hirundo import (
    BiasType,
    EvalRunInfo,
    HuggingFaceTransformersModel,
    LlmBehaviorEval,
    LlmModel,
    ModelOrRun,
    PresetType,
)

llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()

run_id = LlmBehaviorEval.launch_eval_run(
    ModelOrRun.MODEL,
    EvalRunInfo(
        name="Nemotron BBQ bias eval",
        model_id=llm_id,
        preset_type=PresetType.BBQ_BIAS,
        bias_type=BiasType.ALL,
    ),
)

results = LlmBehaviorEval.check_run_by_id(run_id)
print(results.summary_brief)
```

## Quickstart: Dataset QA

### Classification

```python
import json
import os

from hirundo import (
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)

gcp_bucket = StorageGCP(
    bucket_name="cifar100bucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)

test_dataset = QADataset(
    name="TEST-GCP cifar 100 classification dataset",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name="cifar100bucket",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    data_root_url=gcp_bucket.get_url(path="/pytorch-cifar/data"),
    labeling_info=HirundoCSV(
        csv_url=gcp_bucket.get_url(path="/pytorch-cifar/data/cifar100.csv"),
    ),
    classes=cifar100_classes,
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
```

### Object detection

```python
from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGit,
    StorageTypes,
)

git_storage = StorageGit(
    repo=GitRepo(
        name="BDD-100k-validation-dataset",
        repository_url="https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only",
    ),
    branch="main",
)

test_dataset = QADataset(
    name="TEST-HuggingFace-BDD-100k-validation-OD-validation-dataset",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name="BDD-100k-validation-dataset",
        type=StorageTypes.GIT,
        git=git_storage,
    ),
    data_root_url=git_storage.get_url(path="/BDD100K Val from Hirundo.zip/bdd100k"),
    labeling_info=HirundoCSV(
        csv_url=git_storage.get_url(
            path="/BDD100K Val from Hirundo.zip/bdd100k/bdd100k.csv"
        ),
    ),
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
```

## Supported dataset storage

- Amazon S3
- Google Cloud Storage (GCS)
- Git repositories with LFS (GitHub, Hugging Face)

## Further documentation

- Documentation site: [https://docs.hirundo.io/](https://docs.hirundo.io/)
- Example notebooks: [notebooks/](notebooks/)
