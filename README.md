# Hirundo

This package exposes access to Hirundo APIs for dataset QA for Machine Learning.

Dataset QA is currently available for datasets labelled for classification and object detection.

Support dataset storage configs include:

- Google Cloud (GCP) Storage
- Amazon Web Services (AWS) S3
- Git LFS (Large File Storage) repositories (e.g. GitHub or HuggingFace)

Note: This Python package must be used alongside a Hirundo server, either the SaaS platform, a custom VPC deployment or an on-premises installation.

Optimizing a classification dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently `hirundo` requires a CSV file with the following columns (all columns are required):

- `image_path`: The location of the image within the dataset `data_root_url`
- `class_name`: The semantic label, i.e. the class name of the class that the image was annotated as belonging to

And outputs two Pandas DataFrames with the dataset columns as well as:

Suspect DataFrame (filename: `mislabel_suspects.csv`) columns:

- ``suspect_score``: mislabel suspect score
- ``suspect_level``: mislabel suspect level
- ``suspect_rank``: mislabel suspect ranking
- ``suggested_class_name``: suggested semantic label
- ``suggested_class_conf``: suggested semantic label confidence

Errors and warnings DataFrame (filename: `invalid_data.csv`) columns:

   - ``status``: status message (one of ``NO_LABELS`` / ``MISSING_IMAGE`` / ``INVALID_IMAGE``)

Optimizing an object detection (OD) dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``hirundo`` requires a CSV file with the following columns (all columns are required):

- ``image_path``: The location of the image within the dataset ``data_root_url``
- ``object_id``: The ID of the bounding box within the dataset. Used to indicate object suspects
- ``class_name``: Object semantic label, i.e. the class name of the object that was annotated
- ``xmin``: leftmost horizontal pixel coordinate of the object's bounding box
- ``ymin``: uppermost vertical pixel coordinate of the object's bounding box
- ``xmax``: rightmost horizontal pixel coordinate of the object's bounding box
- ``ymax``: lowermost vertical pixel coordinate of the object's bounding box


And outputs two Pandas DataFrames with the dataset columns as well as:

Suspect DataFrame (filename: `mislabel_suspects.csv`) columns:

- ``suspect_score``: object mislabel suspect score
- ``suspect_level``: object mislabel suspect level
- ``suspect_rank``: object mislabel suspect ranking
- ``suggested_class_name``: suggested object semantic label
- ``suggested_class_conf``: suggested object semantic label confidence

Errors and warnings DataFrame (filename: `invalid_data.csv`) columns:
   - ``status``: status message (one of ``NO_LABELS`` / ``MISSING_IMAGE`` / ``INVALID_IMAGE`` / ``INVALID_BBOX`` / ``INVALID_BBOX_SIZE``)

## Installation

You can install the codebase with a simple `pip install hirundo` to install the latest version of this package. If you prefer to install from the Git repository and/or need a specific version or branch, you can simply clone the repository, check out the relevant commit and then run `pip install .` to install that version. A full list of dependencies can be found in `requirements.txt`, but these will be installed automatically by either of these commands.

## Usage

### Unlearning LLM behavior

Make sure to install the `transformers` extra, i.e. `pip install hirundo[transformers]` or `uv pip install hirundo[transformers]` if you have `uv` installed which is much faster than `pip`.

```python
llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()
run_info = BiasRunInfo(
    bias_type=BiasType.ALL,
)
run_id = LlmUnlearningRun.launch(
    llm_id,
    run_info,
)
new_adapter = llm.get_hf_pipeline_for_run(run_id)
```

### Dataset QA

#### Classification example:

```python
from hirundo import (
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageGCP,
    StorageConfig,
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

#### Object detection example:

```python
from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageGit,
    StorageConfig,
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

Note: Currently we only support the main CPython release 3.10, 3.11, 3.12 & 3.13. PyPy support may be introduced in the future.

## Further documentation

To learn more about how to use this library, please visit the [http://docs.hirundo.io/](documentation) or see the [Google Colab examples](https://github.com/Hirundo-io/hirundo-python-sdk/tree/main/notebooks).
