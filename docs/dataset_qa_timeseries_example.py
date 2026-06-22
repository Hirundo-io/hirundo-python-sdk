"""Examples for docs/index.rst literalinclude blocks."""

import json
import os

from hirundo import (
    HirundoCSV,
    LabelingType,
    ModalityType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)

gcp_bucket = StorageGCP(
    bucket_name="timeseriesbucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)

test_dataset = QADataset(
    name="TEST-GCP timeseries classification dataset",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name="timeseriesbucket",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    labeling_info=HirundoCSV(
        csv_url=gcp_bucket.get_url(path="/timeseries/data/metadata.csv"),
    ),
    classes=["normal", "anomaly"],
    modality=ModalityType.TIMESERIES,
    feature_cols=["sensor_a", "sensor_b", "sensor_c"],
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
