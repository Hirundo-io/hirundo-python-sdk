from enum import Enum


class LabelingType(str, Enum):
    """
    Enum indicate what type of labeling is used for the given dataset.
    Supported types are:
    """

    SINGLE_LABEL_CLASSIFICATION = "SingleLabelClassification"
    OBJECT_DETECTION = "ObjectDetection"
    SPEECH_TO_TEXT = "SpeechToText"
    OBJECT_SEGMENTATION = "ObjectSegmentation"
    SEMANTIC_SEGMENTATION = "SemanticSegmentation"
    PANOPTIC_SEGMENTATION = "PanopticSegmentation"


class DatasetMetadataType(str, Enum):
    """
    Enum indicate what type of metadata is provided for the given dataset.
    Supported types are:
    """

    HIRUNDO_CSV = "HirundoCSV"
    COCO = "COCO"
    YOLO = "YOLO"
    KeylabsObjDetImages = "KeylabsObjDetImages"
    KeylabsObjDetVideo = "KeylabsObjDetVideo"
    KeylabsObjSegImages = "KeylabsObjSegImages"
    KeylabsObjSegVideo = "KeylabsObjSegVideo"


class StorageTypes(str, Enum):
    """
    Enum for the different types of storage configs.
    Supported types are:
    """

    S3 = "S3"
    GCP = "GCP"
    # AZURE = "Azure"  TODO: Azure storage config is coming soon
    GIT = "Git"
    LOCAL = "Local"
    """
    Local storage config is only supported for on-premises installations.
    """
    HUGGINGFACE = "HuggingFace"
