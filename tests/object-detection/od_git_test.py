import logging

import pytest
from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGit,
    StorageTypes,
)
from tests.dataset_qa_shared import (
    cleanup,
    dataset_qa_sync_test,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
git_storage = StorageGit(
    repo=GitRepo(
        name=f"BDD-100k-validation-dataset{unique_id}",
        repository_url="https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only",
    ),
    branch="main",
)
test_dataset = QADataset(
    name=f"TEST-HuggingFace-BDD-100k-validation-OD-validation-dataset{unique_id}",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name=f"BDD-100k-validation-dataset{unique_id}",
        type=StorageTypes.GIT,
        git=git_storage,
    ),
    labeling_info=HirundoCSV(
        csv_url=git_storage.get_url(
            path="/BDD100K Val from Hirundo.zip/bdd100k/bdd100k.csv"
        ),
    ),
    data_root_url=git_storage.get_url(path="/BDD100K Val from Hirundo.zip/bdd100k"),
    classes=[
        "traffic light",
        "traffic sign",
        "car",
        "pedestrian",
        "bus",
        "truck",
        "rider",
        "bicycle",
        "motorcycle",
        "train",
        "other vehicle",
        "other person",
        "trailer",
    ],
)


@pytest.fixture(autouse=True)
def cleanup_tests():
    cleanup(test_dataset)
    yield
    cleanup(test_dataset)


def test_dataset_qa():
    full_run = dataset_qa_sync_test(test_dataset, "RUN_OD_GIT_DATA_QA")
    if full_run is not None:
        pass
        # TODO: Add add assertion for result
    else:
        logger.info("Full dataset QA was not run!")
