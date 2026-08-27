"""Examples for docs/index.rst literalinclude blocks."""

import json
import os

from hirundo import (
    HirundoCSV,
    LabelingType,
    ModalityType,
    MultimodalHirundoCSV,
    MultimodalModalityCSV,
    MultimodalModalityType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)

gcp_bucket = StorageGCP(
    bucket_name="multimodalbucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)

test_dataset = QADataset(
    name="TEST-GCP multimodal classification dataset",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name="multimodalbucket",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    labeling_info=MultimodalHirundoCSV(
        modality_csvs=[
            MultimodalModalityCSV(
                modality=MultimodalModalityType.VISION,
                labeling_info=HirundoCSV(
                    csv_url=gcp_bucket.get_url(path="/multimodal/images.csv"),
                ),
                data_root_url=gcp_bucket.get_url(path="/multimodal/images"),
                augmentations=["RandomHorizontalFlip"],
            ),
            MultimodalModalityCSV(
                modality=MultimodalModalityType.TABULAR,
                labeling_info=HirundoCSV(
                    csv_url=gcp_bucket.get_url(path="/multimodal/tabular.csv"),
                ),
                feature_cols=["height", "width", "score"],
            ),
        ],
        alignment_csv_url=gcp_bucket.get_url(path="/multimodal/alignment.csv"),
    ),
    classes=["ok", "defect"],
    modality=ModalityType.MULTIMODAL,
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
