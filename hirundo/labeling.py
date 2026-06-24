import typing
from abc import ABC
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from hirundo._column_options import validate_column_options
from hirundo._urls import HirundoUrl
from hirundo.dataset_enum import DatasetMetadataType


class Metadata(BaseModel, ABC, frozen=True):
    type: DatasetMetadataType


class HirundoCSV(Metadata, frozen=True):
    """
    A dataset metadata file in the Hirundo CSV format
    """

    type: typing.Literal[DatasetMetadataType.HIRUNDO_CSV] = (
        DatasetMetadataType.HIRUNDO_CSV
    )
    csv_url: HirundoUrl
    """
    The URL to access the dataset metadata CSV file.
    e.g. `s3://my-bucket-name/my-folder/my-metadata.csv`, `gs://my-bucket-name/my-folder/my-metadata.csv`,
    or `ssh://my-username@my-repo-name/my-folder/my-metadata.csv`
    (or `file:///datasets/my-folder/my-metadata.csv` if using LOCAL storage type with on-premises installation)
    """


class MultimodalModalityType(str, Enum):
    VISION = "VISION"
    RADAR = "RADAR"
    TABULAR = "TABULAR"
    TIMESERIES = "TIMESERIES"


IMAGE_MULTIMODAL_AUGMENTATIONS = frozenset(
    (
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "RandomPerspective",
        "GaussianNoise",
        "RandomGrayscale",
        "GaussianBlur",
    )
)

TABULAR_MULTIMODAL_AUGMENTATIONS = frozenset(("AddGaussianNoise", "MaskFeatures"))

MULTIMODAL_AUGMENTATIONS_BY_MODALITY = {
    MultimodalModalityType.VISION: IMAGE_MULTIMODAL_AUGMENTATIONS,
    MultimodalModalityType.RADAR: IMAGE_MULTIMODAL_AUGMENTATIONS,
    MultimodalModalityType.TABULAR: TABULAR_MULTIMODAL_AUGMENTATIONS,
    MultimodalModalityType.TIMESERIES: TABULAR_MULTIMODAL_AUGMENTATIONS,
}

MULTIMODAL_AUGMENTATIONS = frozenset(
    augmentation
    for augmentations in MULTIMODAL_AUGMENTATIONS_BY_MODALITY.values()
    for augmentation in augmentations
)


class MultimodalModalityCSV(BaseModel, frozen=True):
    modality: MultimodalModalityType
    labeling_info: HirundoCSV
    data_root_url: HirundoUrl | None = None
    augmentations: list[str] | None = None
    feature_cols: list[str] | None = None
    extra_non_feature_cols: list[str] | None = None

    @field_validator("extra_non_feature_cols", "feature_cols", mode="before")
    @classmethod
    def _normalize_empty_column_options(cls, value):
        if value == []:
            return None
        return value

    @field_validator("augmentations", mode="before")
    @classmethod
    def _validate_known_augmentations(cls, value):
        if value is None:
            return None
        invalid_augmentations = [
            str(getattr(augmentation, "value", augmentation))
            for augmentation in value
            if str(getattr(augmentation, "value", augmentation))
            not in MULTIMODAL_AUGMENTATIONS
        ]
        if invalid_augmentations:
            raise ValueError(
                "Invalid multimodal child augmentations: "
                + ", ".join(invalid_augmentations)
            )
        return value

    @field_validator("modality", mode="before")
    @classmethod
    def _legacy_image_modality(cls, modality: object) -> object:
        if modality == "IMAGE":
            return MultimodalModalityType.VISION
        return modality

    @model_validator(mode="after")
    def _validate_child_options(self) -> "MultimodalModalityCSV":
        validate_column_options(
            feature_cols=self.feature_cols,
            extra_non_feature_cols=self.extra_non_feature_cols,
            modality=self.modality,
            allowed_modalities=frozenset(
                (MultimodalModalityType.TABULAR, MultimodalModalityType.TIMESERIES)
            ),
            unsupported_message=(
                "Multimodal feature column settings are only supported for tabular or timeseries child modalities"
            ),
        )
        if self.augmentations is not None:
            allowed_augmentations = MULTIMODAL_AUGMENTATIONS_BY_MODALITY[self.modality]
            invalid_augmentations = [
                augmentation
                for augmentation in self.augmentations
                if augmentation not in allowed_augmentations
            ]
            if invalid_augmentations:
                raise ValueError(
                    f"Invalid augmentations [{', '.join(invalid_augmentations)}] for {self.modality.value} modality"
                )
        return self


class MultimodalHirundoCSV(Metadata, frozen=True):
    type: typing.Literal[DatasetMetadataType.MULTIMODAL_HIRUNDO_CSV] = (
        DatasetMetadataType.MULTIMODAL_HIRUNDO_CSV
    )
    modality_csvs: list[MultimodalModalityCSV]
    alignment_csv_url: HirundoUrl | None = None

    @model_validator(mode="after")
    def _validate_modality_csvs(self) -> "MultimodalHirundoCSV":
        if len(self.modality_csvs) < 2:
            raise ValueError("Multimodal datasets require at least two modalities")
        modalities = [modality_csv.modality for modality_csv in self.modality_csvs]
        if len(set(modalities)) != len(modalities):
            raise ValueError("Multimodal dataset modalities must be unique")
        return self


class COCO(Metadata, frozen=True):
    """
    A dataset metadata file in the COCO format
    """

    type: typing.Literal[DatasetMetadataType.COCO] = DatasetMetadataType.COCO
    json_url: HirundoUrl
    """
    The URL to access the dataset metadata JSON file.
    e.g. `s3://my-bucket-name/my-folder/my-metadata.json`, `gs://my-bucket-name/my-folder/my-metadata.json`,
    or `ssh://my-username@my-repo-name/my-folder/my-metadata.json`
    (or `file:///datasets/my-folder/my-metadata.json` if using LOCAL storage type with on-premises installation)
    """


class YOLO(Metadata, frozen=True):
    type: typing.Literal[DatasetMetadataType.YOLO] = DatasetMetadataType.YOLO
    data_yaml_url: HirundoUrl | None = None
    labels_dir_url: HirundoUrl


class HuggingFaceAudio(Metadata, frozen=True):
    type: typing.Literal[DatasetMetadataType.HuggingFaceAudio] = (
        DatasetMetadataType.HuggingFaceAudio
    )
    audio_column: str
    text_column: str
    subset: str | None = None
    split: str | None = None


class KeylabsAuth(BaseModel):
    username: str
    password: str
    instance: str


class Keylabs(Metadata, frozen=True):
    project_id: str
    """
    Keylabs project ID.
    """

    labels_dir_url: HirundoUrl
    """
    URL to the directory containing the Keylabs labels.
    """

    with_attributes: bool = True
    """
    Whether to include attributes in the class name.
    """

    project_name: str | None = None
    """
    Keylabs project name (optional; added to output CSV if provided).
    """
    keylabs_auth: KeylabsAuth | None = None
    """
    Keylabs authentication credentials (optional; if provided, used to provide links to each sample).
    """


class KeylabsObjDetImages(Keylabs, frozen=True):
    type: typing.Literal[DatasetMetadataType.KeylabsObjDetImages] = (
        DatasetMetadataType.KeylabsObjDetImages
    )


class KeylabsObjDetVideo(Keylabs, frozen=True):
    type: typing.Literal[DatasetMetadataType.KeylabsObjDetVideo] = (
        DatasetMetadataType.KeylabsObjDetVideo
    )


class KeylabsObjSegImages(Keylabs, frozen=True):
    type: typing.Literal[DatasetMetadataType.KeylabsObjSegImages] = (
        DatasetMetadataType.KeylabsObjSegImages
    )


class KeylabsObjSegVideo(Keylabs, frozen=True):
    type: typing.Literal[DatasetMetadataType.KeylabsObjSegVideo] = (
        DatasetMetadataType.KeylabsObjSegVideo
    )


KeylabsInfo = (
    KeylabsObjDetImages | KeylabsObjDetVideo | KeylabsObjSegImages | KeylabsObjSegVideo
)
"""
The dataset labeling info for Keylabs. The dataset labeling info can be one of the following:
- `DatasetMetadataType.KeylabsObjDetImages`: Indicates that the dataset metadata file is in the Keylabs object detection image format
- `DatasetMetadataType.KeylabsObjDetVideo`: Indicates that the dataset metadata file is in the Keylabs object detection video format
- `DatasetMetadataType.KeylabsObjSegImages`: Indicates that the dataset metadata file is in the Keylabs object segmentation image format
- `DatasetMetadataType.KeylabsObjSegVideo`: Indicates that the dataset metadata file is in the Keylabs object segmentation video format
"""
LabelingInfo = typing.Annotated[
    HirundoCSV | MultimodalHirundoCSV | COCO | YOLO | KeylabsInfo | HuggingFaceAudio,
    Field(discriminator="type"),
]
"""
The dataset labeling info. The dataset labeling info can be one of the following:
- `DatasetMetadataType.HirundoCSV`: Indicates that the dataset metadata file is a CSV file with the Hirundo format
- `DatasetMetadataType.COCO`: Indicates that the dataset metadata file is a JSON file with the COCO format
- `DatasetMetadataType.YOLO`: Indicates that the dataset metadata file is in the YOLO format
- `DatasetMetadataType.KeylabsObjDetImages`: Indicates that the dataset metadata file is in the Keylabs object detection image format
- `DatasetMetadataType.KeylabsObjDetVideo`: Indicates that the dataset metadata file is in the Keylabs object detection video format
- `DatasetMetadataType.KeylabsObjSegImages`: Indicates that the dataset metadata file is in the Keylabs object segmentation image format
- `DatasetMetadataType.KeylabsObjSegVideo`: Indicates that the dataset metadata file is in the Keylabs object segmentation video format

Currently no other formats are supported. Future versions of `hirundo` may support additional formats.
"""
