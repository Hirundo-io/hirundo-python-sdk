import typing
from abc import ABC

from pydantic import BaseModel, Field

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
    HirundoCSV | COCO | YOLO | KeylabsInfo | HuggingFaceAudio,
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
