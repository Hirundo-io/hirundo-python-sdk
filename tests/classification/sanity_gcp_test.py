import json
import logging
import os

import pytest
from hirundo import (
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)
from hirundo.dataset_qa import AugmentationName
from tests.dataset_qa_shared import (
    cleanup,
    dataset_qa_async_test,
    dataset_qa_sync_test,
    get_unique_id,
)

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
gcp_bucket = StorageGCP(
    bucket_name="cifar1bucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)
test_dataset = QADataset(
    name=f"TEST-GCP sanity dataset{unique_id}",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name=f"cifar1bucket{unique_id}",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    labeling_info=HirundoCSV(
        csv_url=gcp_bucket.get_url(path="/pytorch-cifar/data/cifar1.csv"),
    ),
    data_root_url=gcp_bucket.get_url(path="/pytorch-cifar/data"),
    classes=[
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    augmentations=[
        AugmentationName.RANDOM_HORIZONTAL_FLIP,
        AugmentationName.RANDOM_VERTICAL_FLIP,
        AugmentationName.GAUSSIAN_NOISE,
    ],
)


@pytest.fixture(autouse=True)
def cleanup_tests():
    cleanup(test_dataset)
    yield
    cleanup(test_dataset)


def test_dataset_qa():
    full_run = dataset_qa_sync_test(
        test_dataset,
        sanity=True,
        alternative_env="RUN_CLASSIFICATION_GCP_SANITY_DATA_QA",
    )
    if full_run is not None:
        assert full_run.warnings_and_errors is not None
        assert full_run.warnings_and_errors.shape[0] == 0
        assert full_run.suspects is not None
        assert full_run.suspects.shape[0] >= 5_000
    else:
        logger.info("Full dataset QA was not run!")


@pytest.mark.asyncio
async def test_async_dataset_qa():
    await dataset_qa_async_test(test_dataset, "RUN_CLASSIFICATION_GCP_SANITY_DATA_QA")
