import logging
import os

import pytest
from hirundo import (
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageS3,
    StorageTypes,
)
from tests.dataset_qa_shared import (
    cleanup,
    dataset_qa_sync_test,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
s3_bucket = StorageS3(
    bucket_url="s3://cifar10bucket",
    region_name="us-east-2",
    access_key_id=os.environ["AWS_ACCESS_KEY"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
test_dataset = QADataset(
    name=f"TEST-AWS cifar10 classification dataset{unique_id}",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name=f"cifar10bucket{unique_id}",
        type=StorageTypes.S3,
        s3=s3_bucket,
    ),
    labeling_info=HirundoCSV(
        csv_url=s3_bucket.get_url(path="/pytorch-cifar/data/cifar10.csv"),
    ),
    data_root_url=s3_bucket.get_url(path="/pytorch-cifar/data"),
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
)


@pytest.fixture(autouse=True)
def cleanup_tests():
    cleanup(test_dataset)
    yield
    cleanup(test_dataset)


def test_dataset_qa():
    full_run = dataset_qa_sync_test(test_dataset, "RUN_CLASSIFICATION_AWS_DATA_QA")
    if full_run is not None:
        assert full_run.warnings_and_errors is not None
        assert full_run.warnings_and_errors.shape[0] == 0
        # TODO: Add more assertions for results
    else:
        logger.info("Full dataset QA was not run!")
