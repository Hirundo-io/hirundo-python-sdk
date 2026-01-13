import json
import logging
import os

import pytest
from hirundo import (
    COCO,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGCP,
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
gcp_bucket = StorageGCP(
    bucket_name="sama-coco-bucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)
test_dataset = QADataset(
    name=f"TEST-GCP sanity COCO dataset{unique_id}",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name=f"sama-coco-{unique_id}",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    labeling_info=COCO(
        json_url=gcp_bucket.get_url(
            path="/sama-coco.zip/sama-coco/validation/labels.json"
        ),
    ),
    data_root_url=gcp_bucket.get_url(path="/sama-coco.zip/sama-coco/validation/data"),
)


@pytest.fixture(autouse=True)
def cleanup_tests():
    cleanup(test_dataset)
    yield
    cleanup(test_dataset)


def test_dataset_qa():
    full_run = dataset_qa_sync_test(
        test_dataset,
        alternative_env="RUN_COCO_OD_GCP_SANITY_DATA_QA",
    )
    if full_run is not None:
        assert full_run.warnings_and_errors is not None
        assert full_run.warnings_and_errors.shape[0] == 0
        assert full_run.suspects is not None
        assert full_run.suspects.shape[0] >= 30_000
        # TODO: Add more assertions for results
    else:
        logger.info("Full dataset QA was not run!")


@pytest.mark.asyncio
async def test_async_dataset_qa():
    await dataset_qa_async_test(test_dataset, "RUN_COCO_OD_GCP_SANITY_DATA_QA")
