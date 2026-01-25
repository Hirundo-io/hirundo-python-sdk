import logging
import os

import pytest
from hirundo import (
    HirundoCSV,
    LabelingType,
    ObjectDetectionRunArgs,
    QADataset,
    StorageConfig,
    StorageS3,
    StorageTypes,
)
from tests.dataset_qa_shared import (
    cleanup,
    dataset_qa_async_test,
    dataset_qa_sync_test,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
s3_bucket = StorageS3(
    bucket_url="s3://hirundo-open-source-datasets",
    region_name="il-central-1",
    access_key_id=os.environ["AWS_ACCESS_KEY"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
test_dataset = QADataset(
    name=f"TEST-AWS-BDD-100k-subset-1000-OD-dataset{unique_id}",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name=f"AWS-open-source-datasets-sanity{unique_id}",
        type=StorageTypes.S3,
        s3=s3_bucket,
    ),
    labeling_info=HirundoCSV(
        csv_url=s3_bucket.get_url(
            path="/bdd100k_subset_1000_hirundo.zip/bdd100k/bdd100k.csv"
        ),
    ),
    data_root_url=s3_bucket.get_url(path="/bdd100k_subset_1000_hirundo.zip/bdd100k"),
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
    full_run = dataset_qa_sync_test(
        test_dataset,
        sanity=True,
        alternative_env="RUN_OD_AWS_SANITY_DATA_QA",
        run_args=ObjectDetectionRunArgs(
            upsample=True,
            min_abs_bbox_size=11,
            min_abs_bbox_area=121,
        ),
    )
    if full_run is not None:
        assert full_run.warnings_and_errors is not None
        assert full_run.warnings_and_errors.shape[0] >= 120
        logger.info(
            "Warnings and errors count: %s", full_run.warnings_and_errors.shape[0]
        )
        assert full_run.suspects is not None
        assert full_run.suspects.shape[0] == 1_107
        # TODO: Add more assertions for results
    else:
        logger.info("Full dataset QA was not run!")


@pytest.mark.asyncio
async def test_async_dataset_qa():
    await dataset_qa_async_test(test_dataset, "RUN_AWS_OD_SANITY_DATA_QA")
