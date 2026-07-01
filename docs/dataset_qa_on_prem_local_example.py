"""On-premises local storage Dataset QA example."""

from pydantic_core import Url

from hirundo import (
    HirundoCSV,
    LabelingType,
    ModalityType,
    MultimodalHirundoCSV,
    MultimodalModalityCSV,
    MultimodalModalityType,
    QADataset,
    StorageConfig,
)

local_storage = StorageConfig.get_default_local()

test_dataset = QADataset(
    name="on-prem local multimodal dataset",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=local_storage,
    labeling_info=MultimodalHirundoCSV(
        modality_csvs=[
            MultimodalModalityCSV(
                modality=MultimodalModalityType.VISION,
                labeling_info=HirundoCSV(
                    csv_url=Url("file:///datasets/multimodal/images.csv"),
                ),
                data_root_url=Url("file:///datasets/multimodal/images"),
                augmentations=["RandomHorizontalFlip"],
            ),
            MultimodalModalityCSV(
                modality=MultimodalModalityType.TABULAR,
                labeling_info=HirundoCSV(
                    csv_url=Url("file:///datasets/multimodal/tabular.csv"),
                ),
                feature_cols=["height", "width", "score"],
            ),
        ],
        alignment_csv_url=Url("file:///datasets/multimodal/alignment.csv"),
    ),
    classes=["ok", "defect"],
    modality=ModalityType.MULTIMODAL,
)

test_dataset.run_qa()
