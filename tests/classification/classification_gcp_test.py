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
from tests.classification.cifar100_classes import cifar100_classes
from tests.dataset_qa_shared import (
    cleanup,
    dataset_qa_sync_test,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()
gcp_bucket = StorageGCP(
    bucket_name="cifar100bucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)
test_dataset = QADataset(
    name=f"TEST-GCP cifar 100 classification dataset{unique_id}",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name=f"cifar100bucket{unique_id}",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    data_root_url=gcp_bucket.get_url(path="/pytorch-cifar/data"),
    labeling_info=HirundoCSV(
        csv_url=gcp_bucket.get_url(path="/pytorch-cifar/data/cifar100.csv"),
    ),
    classes=cifar100_classes,
)


@pytest.fixture(autouse=True)
def cleanup_tests():
    cleanup(test_dataset)
    yield
    cleanup(test_dataset)


def test_dataset_qa():
    full_run = dataset_qa_sync_test(test_dataset, "RUN_CLASSIFICATION_GCP_DATA_QA")
    if full_run is not None:
        assert full_run.warnings_and_errors is not None
        assert full_run.warnings_and_errors.shape[0] == 0
        # TODO: Add more assertions for results
    else:
        logger.info("Full dataset QA was not run!")
