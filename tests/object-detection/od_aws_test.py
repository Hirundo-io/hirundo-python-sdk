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
    get_unique_id,
)

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
s3_bucket = StorageS3(
    bucket_url="s3://hirundo-open-source-datasets",
    region_name="il-central-1",
    access_key_id=os.environ["AWS_ACCESS_KEY"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
test_dataset = QADataset(
    name=f"TEST-AWS-BDD-100k-validation-OD-dataset{unique_id}",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name=f"AWS-open-source-datasets{unique_id}",
        type=StorageTypes.S3,
        s3=s3_bucket,
    ),
    labeling_info=HirundoCSV(
        csv_url=s3_bucket.get_url(path="/bdd100k_val_hirundo.zip/bdd100k/bdd100k.csv"),
    ),
    data_root_url=s3_bucket.get_url(path="/bdd100k_val_hirundo.zip/bdd100k"),
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
    full_run = dataset_qa_sync_test(test_dataset, "RUN_AWS_OD_DATA_QA")
    if full_run is not None:
        pass
        # TODO: Add add assertion for result
    else:
        logger.info("Full dataset QA was not run!")
