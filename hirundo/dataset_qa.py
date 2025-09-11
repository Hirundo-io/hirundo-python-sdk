import datetime
import json
import typing
from collections.abc import AsyncGenerator, Generator
from enum import Enum
from typing import overload

import httpx
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hirundo._constraints import validate_labeling_info, validate_url
from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo._urls import HirundoUrl
from hirundo.dataset_enum import DatasetMetadataType, LabelingType
from hirundo.dataset_qa_results import DatasetQAResults
from hirundo.labeling import YOLO, LabelingInfo
from hirundo.logger import get_logger
from hirundo.storage import ResponseStorageConfig, StorageConfig
from hirundo.unzip import download_and_extract_zip

logger = get_logger(__name__)


class HirundoError(Exception):
    """
    Custom exception used to indicate errors in `hirundo` dataset QA runs
    """

    pass


MAX_RETRIES = 200  # Max 200 retries for HTTP SSE connection


class RunStatus(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    AWAITING_MANUAL_APPROVAL = "AWAITING MANUAL APPROVAL"
    REVOKED = "REVOKED"
    REJECTED = "REJECTED"
    RETRY = "RETRY"


STATUS_TO_TEXT_MAP = {
    RunStatus.STARTED.value: "Dataset QA run in progress. Downloading dataset",
    RunStatus.PENDING.value: "Dataset QA run queued and not yet started",
    RunStatus.SUCCESS.value: "Dataset QA run completed successfully",
    RunStatus.FAILURE.value: "Dataset QA run failed",
    RunStatus.AWAITING_MANUAL_APPROVAL.value: "Awaiting manual approval",
    RunStatus.RETRY.value: "Dataset QA run failed. Retrying",
    RunStatus.REVOKED.value: "Dataset QA run was cancelled",
    RunStatus.REJECTED.value: "Dataset QA run was rejected",
}
STATUS_TO_PROGRESS_MAP = {
    RunStatus.STARTED.value: 0.0,
    RunStatus.PENDING.value: 0.0,
    RunStatus.SUCCESS.value: 100.0,
    RunStatus.FAILURE.value: 100.0,
    RunStatus.AWAITING_MANUAL_APPROVAL.value: 100.0,
    RunStatus.RETRY.value: 0.0,
    RunStatus.REVOKED.value: 100.0,
    RunStatus.REJECTED.value: 0.0,
}


class ClassificationRunArgs(BaseModel):
    image_size: typing.Optional[tuple[int, int]] = (224, 224)
    """
    Size (width, height) to which to resize classification images.
    It is recommended to keep this value at (224, 224) unless your classes are differentiated by very small differences.
    """
    upsample: typing.Optional[bool] = False
    """
    Whether to upsample the dataset to attempt to balance the classes.
    """


class ObjectDetectionRunArgs(ClassificationRunArgs):
    min_abs_bbox_size: typing.Optional[int] = None
    """
    Minimum valid size (in pixels) of a bounding box to keep it in the dataset for QA.
    """
    min_abs_bbox_area: typing.Optional[int] = None
    """
    Minimum valid absolute area (in pixels²) of a bounding box to keep it in the dataset for QA.
    """
    min_rel_bbox_size: typing.Optional[float] = None
    """
    Minimum valid size (as a fraction of both image height and width) for a bounding box
    to keep it in the dataset for QA, relative to the corresponding dimension size,
    i.e. if the bounding box is 10% of the image width and 5% of the image height, it will be kept if this value is 0.05, but not if the
    value is 0.06 (since both width and height are checked).
    """
    min_rel_bbox_area: typing.Optional[float] = None
    """
    Minimum valid relative area (as a fraction of the image area) of a bounding box to keep it in the dataset for QA.
    """
    crop_ratio: typing.Optional[float] = None
    """
    Ratio of the bounding box to crop.
    Change this value at your own risk. It is recommended to keep it at 1.0 unless you know what you are doing.
    """
    add_mask_channel: typing.Optional[bool] = None
    """
    Whether to add a mask channel to the image.
    Change at your own risk. It is recommended to keep it at False unless you know what you are doing.
    """


RunArgs = typing.Union[ClassificationRunArgs, ObjectDetectionRunArgs]


class AugmentationName(str, Enum):
    RANDOM_HORIZONTAL_FLIP = "RandomHorizontalFlip"
    RANDOM_VERTICAL_FLIP = "RandomVerticalFlip"
    RANDOM_ROTATION = "RandomRotation"
    RANDOM_PERSPECTIVE = "RandomPerspective"
    GAUSSIAN_NOISE = "GaussianNoise"
    RANDOM_GRAYSCALE = "RandomGrayscale"
    GAUSSIAN_BLUR = "GaussianBlur"


class Domain(str, Enum):
    RADAR = "RADAR"
    VISION = "VISION"
    SPEECH = "SPEECH"
    TABULAR = "TABULAR"


DOMAIN_TO_SUPPORTED_LABELING_TYPES = {
    Domain.RADAR: [
        LabelingType.SINGLE_LABEL_CLASSIFICATION,
        LabelingType.OBJECT_DETECTION,
    ],
    Domain.VISION: [
        LabelingType.SINGLE_LABEL_CLASSIFICATION,
        LabelingType.OBJECT_DETECTION,
        LabelingType.OBJECT_SEGMENTATION,
        LabelingType.SEMANTIC_SEGMENTATION,
        LabelingType.PANOPTIC_SEGMENTATION,
    ],
    Domain.SPEECH: [LabelingType.SPEECH_TO_TEXT],
    Domain.TABULAR: [LabelingType.SINGLE_LABEL_CLASSIFICATION],
}


class QADataset(BaseModel):
    id: typing.Optional[int] = Field(default=None)
    """
    The ID of the dataset created on the server.
    """
    name: str
    """
    The name of the dataset. Used to identify it amongst the list of datasets
    belonging to your organization in `hirundo`.
    """
    labeling_type: LabelingType
    """
    Indicates the labeling type of the dataset. The labeling type can be one of the following:
    - `LabelingType.SINGLE_LABEL_CLASSIFICATION`: Indicates that the dataset is for classification tasks
    - `LabelingType.OBJECT_DETECTION`: Indicates that the dataset is for object detection tasks
    - `LabelingType.SPEECH_TO_TEXT`: Indicates that the dataset is for speech-to-text tasks
    """
    language: typing.Optional[str] = None
    """
    Language of the Speech-to-Text audio dataset. This is required for Speech-to-Text datasets.
    """
    storage_config_id: typing.Optional[int] = None
    """
    The ID of the storage config used to store the dataset and metadata.
    """
    storage_config: typing.Optional[
        typing.Union[StorageConfig, ResponseStorageConfig]
    ] = None
    """
    The `StorageConfig` instance to link to.
    """
    data_root_url: HirundoUrl
    """
    URL for data (e.g. images) within the `StorageConfig` instance,
    e.g. `s3://my-bucket-name/my-images-folder`, `gs://my-bucket-name/my-images-folder`,
    or `ssh://my-username@my-repo-name/my-images-folder`
    (or `file:///datasets/my-images-folder` if using LOCAL storage type with on-premises installation)

    Note: All CSV `image_path` entries in the metadata file should be relative to this folder.
    """

    classes: typing.Optional[list[str]] = None
    """
    A full list of possible classes used in classification / object detection.
    It is currently required for clarity and performance.
    """
    labeling_info: typing.Union[LabelingInfo, list[LabelingInfo]]

    augmentations: typing.Optional[list[AugmentationName]] = None
    """
    Used to define which augmentations are apply to a vision dataset.
    For audio datasets, this field is ignored.
    If no value is provided, all augmentations are applied to vision datasets.
    """
    domain: Domain = Domain.VISION
    """
    Used to define the domain of the dataset.
    Defaults to Image.
    """

    run_id: typing.Optional[str] = Field(default=None, init=False)
    """
    The ID of the Dataset QA run created on the server.
    """

    status: typing.Optional[RunStatus] = None

    @model_validator(mode="after")
    def validate_dataset(self):
        if self.domain not in DOMAIN_TO_SUPPORTED_LABELING_TYPES:
            raise ValueError(
                f"Domain {self.domain} is not supported. Supported domains are: {list(DOMAIN_TO_SUPPORTED_LABELING_TYPES.keys())}"
            )
        if self.labeling_type not in DOMAIN_TO_SUPPORTED_LABELING_TYPES[self.domain]:
            raise ValueError(
                f"Labeling type {self.labeling_type} is not supported for domain {self.domain}. Supported labeling types are: {DOMAIN_TO_SUPPORTED_LABELING_TYPES[self.domain]}"
            )
        if self.storage_config is None and self.storage_config_id is None:
            raise ValueError(
                "No dataset storage has been provided. Provide one via `storage_config` or `storage_config_id`"
            )
        elif self.storage_config is not None and self.storage_config_id is not None:
            raise ValueError(
                "Both `storage_config` and `storage_config_id` have been provided. Pick one."
            )
        if self.labeling_type == LabelingType.SPEECH_TO_TEXT and self.language is None:
            raise ValueError("Language is required for Speech-to-Text datasets.")
        elif (
            self.labeling_type != LabelingType.SPEECH_TO_TEXT
            and self.language is not None
        ):
            raise ValueError("Language is only allowed for Speech-to-Text datasets.")
        if (
            not isinstance(self.labeling_info, list)
            and self.labeling_info.type == DatasetMetadataType.YOLO
            and isinstance(self.labeling_info, YOLO)
            and (
                self.labeling_info.data_yaml_url is not None
                and self.classes is not None
            )
        ) or (
            isinstance(self.labeling_info, list)
            and self.classes is not None
            and any(
                isinstance(info, YOLO) and info.data_yaml_url is not None
                for info in self.labeling_info
            )
        ):
            raise ValueError(
                "Only one of `classes` or `labeling_info.data_yaml_url` should be provided for YOLO datasets"
            )
        if self.storage_config:
            validate_labeling_info(
                self.labeling_type, self.labeling_info, self.storage_config
            )
        if self.data_root_url and self.storage_config:
            validate_url(self.data_root_url, self.storage_config)
        return self

    @staticmethod
    def get_by_id(dataset_id: int) -> "QADataset":
        """
        Get a `QADataset` instance from the server by its ID

        Args:
            dataset_id: The ID of the `QADataset` instance to get
        """
        response = requests.get(
            f"{API_HOST}/dataset-qa/dataset/{dataset_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        dataset = response.json()
        return QADataset(**dataset)

    @staticmethod
    def get_by_name(name: str) -> "QADataset":
        """
        Get a `QADataset` instance from the server by its name

        Args:
            name: The name of the `QADataset` instance to get
        """
        response = requests.get(
            f"{API_HOST}/dataset-qa/dataset/by-name/{name}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        dataset = response.json()
        return QADataset(**dataset)

    @staticmethod
    def list_datasets(
        organization_id: typing.Optional[int] = None,
    ) -> list["QADatasetOut"]:
        """
        Lists all the datasets created by user's default organization
        or the `organization_id` passed

        Args:
            organization_id: The ID of the organization to list the datasets for.
        """
        response = requests.get(
            f"{API_HOST}/dataset-qa/dataset/",
            params={"dataset_organization_id": organization_id},
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        datasets = response.json()
        return [
            QADatasetOut(
                **ds,
            )
            for ds in datasets
        ]

    @staticmethod
    def list_runs(
        organization_id: typing.Optional[int] = None,
    ) -> list["DataQARunOut"]:
        """
        Lists all the `QADataset` instances created by user's default organization
        or the `organization_id` passed
        Note: The return type is `list[dict]` and not `list[QADataset]`

        Args:
            organization_id: The ID of the organization to list the datasets for.
        """
        response = requests.get(
            f"{API_HOST}/dataset-qa/run/list",
            params={"dataset_organization_id": organization_id},
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        runs = response.json()
        return [
            DataQARunOut(
                **run,
            )
            for run in runs
        ]

    @staticmethod
    def delete_by_id(dataset_id: int) -> None:
        """
        Deletes a `QADataset` instance from the server by its ID

        Args:
            dataset_id: The ID of the `QADataset` instance to delete
        """
        response = requests.delete(
            f"{API_HOST}/dataset-qa/dataset/{dataset_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        logger.info("Deleted dataset with ID: %s", dataset_id)

    def delete(self, storage_config=True) -> None:
        """
        Deletes the active `QADataset` instance from the server.
        It can only be used on a `QADataset` instance that has been created.

        Args:
            storage_config: If True, the `QADataset`'s `StorageConfig` will also be deleted

        Note: If `storage_config` is not set to `False` then the `storage_config_id` must be set
        This can either be set manually or by creating the `StorageConfig` instance via the `QADataset`'s
        `create` method
        """
        if storage_config:
            if not self.storage_config_id:
                raise ValueError("No storage config has been created")
            StorageConfig.delete_by_id(self.storage_config_id)
        if not self.id:
            raise ValueError("No dataset has been created")
        self.delete_by_id(self.id)

    def create(
        self,
        organization_id: typing.Optional[int] = None,
        replace_if_exists: bool = False,
    ) -> int:
        """
        Create a `QADataset` instance on the server.
        If the `storage_config_id` field is not set, the storage config will also be created and the field will be set.

        Args:
            organization_id: The ID of the organization to create the dataset for.
            replace_if_exists: If True, the dataset will be replaced if it already exists
                (this is determined by a dataset of the same name in the same organization).

        Returns:
            The ID of the created `QADataset` instance
        """
        if self.storage_config is None and self.storage_config_id is None:
            raise ValueError("No dataset storage has been provided")
        elif self.storage_config and self.storage_config_id is None:
            if isinstance(self.storage_config, ResponseStorageConfig):
                self.storage_config_id = self.storage_config.id
            elif isinstance(self.storage_config, StorageConfig):
                self.storage_config_id = self.storage_config.create(
                    replace_if_exists=replace_if_exists,
                )
        elif (
            self.storage_config is not None
            and self.storage_config_id is not None
            and (
                not isinstance(self.storage_config, ResponseStorageConfig)
                or self.storage_config.id != self.storage_config_id
            )
        ):
            raise ValueError(
                "Both `storage_config` and `storage_config_id` have been provided. Storage config IDs do not match."
            )
        model_dict = self.model_dump(mode="json")
        # ⬆️ Get dict of model fields from Pydantic model instance
        dataset_response = requests.post(
            f"{API_HOST}/dataset-qa/dataset/",
            json={
                **{k: model_dict[k] for k in model_dict.keys() - {"storage_config"}},
                "organization_id": organization_id,
                "replace_if_exists": replace_if_exists,
            },
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(dataset_response)
        self.id = dataset_response.json()["id"]
        if not self.id:
            raise HirundoError("An error ocurred while trying to create the dataset")
        logger.info("Created dataset with ID: %s", self.id)
        return self.id

    @staticmethod
    def launch_qa_run(
        dataset_id: int,
        organization_id: typing.Optional[int] = None,
        run_args: typing.Optional[RunArgs] = None,
    ) -> str:
        """
        Run the dataset QA process on the server using the dataset with the given ID
        i.e. `dataset_id`.

        Args:
            dataset_id: The ID of the dataset to run QA on.

        Returns:
            ID of the run (`run_id`).
        """
        run_info = {}
        if organization_id:
            run_info["organization_id"] = organization_id
        if run_args:
            run_info["run_args"] = run_args.model_dump(mode="json")
        run_response = requests.post(
            f"{API_HOST}/dataset-qa/run/{dataset_id}",
            json=run_info if len(run_info) > 0 else None,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        return run_response.json()["run_id"]

    def _validate_run_args(self, run_args: RunArgs) -> None:
        if self.labeling_type == LabelingType.SPEECH_TO_TEXT:
            raise Exception("Speech to text cannot have `run_args` set")
        if (
            self.labeling_type != LabelingType.OBJECT_DETECTION
            and isinstance(run_args, ObjectDetectionRunArgs)
            and any(
                (
                    run_args.min_abs_bbox_size != 0,
                    run_args.min_abs_bbox_area != 0,
                    run_args.min_rel_bbox_size != 0,
                    run_args.min_rel_bbox_area != 0,
                )
            )
        ):
            raise Exception(
                "Cannot set `min_abs_bbox_size`, `min_abs_bbox_area`, "
                + "`min_rel_bbox_size`, or `min_rel_bbox_area` for "
                + f"labeling type {self.labeling_type}"
            )

    def run_qa(
        self,
        organization_id: typing.Optional[int] = None,
        replace_dataset_if_exists: bool = False,
        run_args: typing.Optional[RunArgs] = None,
    ) -> str:
        """
        If the dataset was not created on the server yet, it is created.
        Run the dataset QA process on the server using the active `QADataset` instance

        Args:
            organization_id: The ID of the organization to run the QA for.
            replace_dataset_if_exists: If True, the dataset will be replaced if it already exists
                (this is determined by a dataset of the same name in the same organization).
            run_args: The run arguments to use for the QA run

        Returns:
            An ID of the run (`run_id`) and stores that `run_id` on the instance
        """
        try:
            if not self.id:
                self.id = self.create(replace_if_exists=replace_dataset_if_exists)
            if run_args is not None:
                self._validate_run_args(run_args)
            run_id = self.launch_qa_run(self.id, organization_id, run_args)
            self.run_id = run_id
            logger.info("Started the run with ID: %s", run_id)
            return run_id
        except requests.HTTPError as error:
            try:
                content = error.response.json()
                logger.error(
                    "HTTP Error! Status code:",
                    error.response.status_code,
                    "Content:",
                    content,
                )
            except Exception:
                content = error.response.text
            raise HirundoError(
                f"Unable to start the run. Status code: {error.response.status_code} Content: {content}"
            ) from error
        except Exception as error:
            raise HirundoError(f"Unable to start the run: {error}") from error

    def clean_ids(self):
        """
        Reset `dataset_id`, `storage_config_id`, and `run_id` values on the instance to default value of `None`
        """
        self.storage_config_id = None
        self.id = None
        self.run_id = None

    @staticmethod
    def _check_run_by_id(run_id: str, retry=0) -> Generator[dict, None, None]:
        if retry > MAX_RETRIES:
            raise HirundoError("Max retries reached")
        last_event = None
        with httpx.Client(timeout=httpx.Timeout(None, connect=5.0)) as client:
            for sse in iter_sse_retrying(
                client,
                "GET",
                f"{API_HOST}/dataset-qa/run/{run_id}",
                headers=get_headers(),
            ):
                if sse.event == "ping":
                    continue
                logger.debug(
                    "[SYNC] received event: %s with data: %s and ID: %s and retry: %s",
                    sse.event,
                    sse.data,
                    sse.id,
                    sse.retry,
                )
                last_event = json.loads(sse.data)
                if not last_event:
                    continue
                if "data" in last_event:
                    data = last_event["data"]
                else:
                    if "detail" in last_event:
                        raise HirundoError(last_event["detail"])
                    elif "reason" in last_event:
                        raise HirundoError(last_event["reason"])
                    else:
                        raise HirundoError("Unknown error")
                yield data
        if not last_event or last_event["data"]["state"] == RunStatus.PENDING.value:
            QADataset._check_run_by_id(run_id, retry + 1)

    @staticmethod
    def _handle_failure(iteration: dict):
        if iteration["result"]:
            raise HirundoError(f"QA run failed with error: {iteration['result']}")
        else:
            raise HirundoError("QA run failed with an unknown error in _handle_failure")

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: typing.Literal[True]
    ) -> typing.Optional[DatasetQAResults]: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: typing.Literal[False] = False
    ) -> DatasetQAResults: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool
    ) -> typing.Optional[DatasetQAResults]: ...

    @staticmethod
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool = False
    ) -> typing.Optional[DatasetQAResults]:
        """
        Check the status of a run given its ID

        Args:
            run_id: The `run_id` produced by a `run_qa` call
            stop_on_manual_approval: If True, the function will return `None` if the run is awaiting manual approval

        Returns:
            A DatasetQAResults object with the results of the QA run

        Raises:
            HirundoError: If the maximum number of retries is reached or if the run fails
        """
        logger.debug("Checking run with ID: %s", run_id)
        with logging_redirect_tqdm():
            t = tqdm(total=100.0)
            for iteration in QADataset._check_run_by_id(run_id):
                if iteration["state"] in STATUS_TO_PROGRESS_MAP:
                    t.set_description(STATUS_TO_TEXT_MAP[iteration["state"]])
                    t.n = STATUS_TO_PROGRESS_MAP[iteration["state"]]
                    logger.debug("Setting progress to %s", t.n)
                    t.refresh()
                    if iteration["state"] in [
                        RunStatus.FAILURE.value,
                        RunStatus.REJECTED.value,
                        RunStatus.REVOKED.value,
                    ]:
                        logger.error(
                            "State is failure, rejected, or revoked: %s",
                            iteration["state"],
                        )
                        QADataset._handle_failure(iteration)
                    elif iteration["state"] == RunStatus.SUCCESS.value:
                        t.close()
                        zip_temporary_url = iteration["result"]
                        logger.debug("QA run completed. Downloading results")

                        return download_and_extract_zip(
                            run_id,
                            zip_temporary_url,
                        )
                    elif (
                        iteration["state"] == RunStatus.AWAITING_MANUAL_APPROVAL.value
                        and stop_on_manual_approval
                    ):
                        t.close()
                        return None
                elif iteration["state"] is None:
                    if (
                        iteration["result"]
                        and isinstance(iteration["result"], dict)
                        and iteration["result"]["result"]
                        and isinstance(iteration["result"]["result"], str)
                    ):
                        result_info = iteration["result"]["result"].split(":")
                        if len(result_info) > 1:
                            stage = result_info[0]
                            current_progress_percentage = float(
                                result_info[1].removeprefix(" ").removesuffix("% done")
                            )
                        elif len(result_info) == 1:
                            stage = result_info[0]
                            current_progress_percentage = t.n  # Keep the same progress
                        else:
                            stage = "Unknown progress state"
                            current_progress_percentage = t.n  # Keep the same progress
                        desc = (
                            "QA run completed. Uploading results"
                            if current_progress_percentage == 100.0
                            else stage
                        )
                        t.set_description(desc)
                        t.n = current_progress_percentage
                        logger.debug("Setting progress to %s", t.n)
                        t.refresh()
        raise HirundoError("QA run failed with an unknown error in check_run_by_id")

    @overload
    def check_run(
        self, stop_on_manual_approval: typing.Literal[True]
    ) -> typing.Optional[DatasetQAResults]: ...

    @overload
    def check_run(
        self, stop_on_manual_approval: typing.Literal[False] = False
    ) -> DatasetQAResults: ...

    def check_run(
        self, stop_on_manual_approval: bool = False
    ) -> typing.Optional[DatasetQAResults]:
        """
        Check the status of the current active instance's run.

        Returns:
            A pandas DataFrame with the results of the QA run

        """
        if not self.run_id:
            raise ValueError("No run has been started")
        return self.check_run_by_id(self.run_id, stop_on_manual_approval)

    @staticmethod
    async def acheck_run_by_id(run_id: str, retry=0) -> AsyncGenerator[dict, None]:
        """
        Async version of :func:`check_run_by_id`

        Check the status of a run given its ID.

        This generator will produce values to show progress of the run.

        Args:
            run_id: The `run_id` produced by a `run_qa` call
            retry: A number used to track the number of retries to limit re-checks. *Do not* provide this value manually.

        Yields:
            Each event will be a dict, where:
            - `"state"` is PENDING, STARTED, RETRY, FAILURE or SUCCESS
            - `"result"` is a string describing the progress as a percentage for a PENDING state, or the error for a FAILURE state or the results for a SUCCESS state

        """
        logger.debug("Checking run with ID: %s", run_id)
        if retry > MAX_RETRIES:
            raise HirundoError("Max retries reached")
        last_event = None
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=5.0)
        ) as client:
            async_iterator = await aiter_sse_retrying(
                client,
                "GET",
                f"{API_HOST}/dataset-qa/run/{run_id}",
                headers=get_headers(),
            )
            async for sse in async_iterator:
                if sse.event == "ping":
                    continue
                logger.debug(
                    "[ASYNC] Received event: %s with data: %s and ID: %s and retry: %s",
                    sse.event,
                    sse.data,
                    sse.id,
                    sse.retry,
                )
                last_event = json.loads(sse.data)
                yield last_event["data"]
        if not last_event or last_event["data"]["state"] == RunStatus.PENDING.value:
            QADataset.acheck_run_by_id(run_id, retry + 1)

    async def acheck_run(self) -> AsyncGenerator[dict, None]:
        """
        Async version of :func:`check_run`

        Check the status of the current active instance's run.

        This generator will produce values to show progress of the run.

        Yields:
            Each event will be a dict, where:
            - `"state"` is PENDING, STARTED, RETRY, FAILURE or SUCCESS
            - `"result"` is a string describing the progress as a percentage for a PENDING state, or the error for a FAILURE state or the results for a SUCCESS state

        """
        if not self.run_id:
            raise ValueError("No run has been started")
        async for iteration in self.acheck_run_by_id(self.run_id):
            yield iteration

    @staticmethod
    def cancel_by_id(run_id: str) -> None:
        """
        Cancel the dataset QA run for the given `run_id`.

        Args:
            run_id: The ID of the run to cancel
        """
        logger.info("Cancelling run with ID: %s", run_id)
        response = requests.delete(
            f"{API_HOST}/dataset-qa/run/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def cancel(self) -> None:
        """
        Cancel the current active instance's run.
        """
        if not self.run_id:
            raise ValueError("No run has been started")
        self.cancel_by_id(self.run_id)

    @staticmethod
    def archive_run_by_id(run_id: str) -> None:
        """
        Archive the dataset QA run for the given `run_id`.

        Args:
            run_id: The ID of the run to archive
        """
        logger.info("Archiving run with ID: %s", run_id)
        response = requests.patch(
            f"{API_HOST}/dataset-qa/run/archive/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def archive(self) -> None:
        """
        Archive the current active instance's run.
        """
        if not self.run_id:
            raise ValueError("No run has been started")
        self.archive_run_by_id(self.run_id)


class QADatasetOut(BaseModel):
    id: int

    name: str
    labeling_type: LabelingType

    storage_config: ResponseStorageConfig

    data_root_url: HirundoUrl

    classes: typing.Optional[list[str]] = None
    labeling_info: typing.Union[LabelingInfo, list[LabelingInfo]]

    organization_id: typing.Optional[int]
    creator_id: typing.Optional[int]
    created_at: datetime.datetime
    updated_at: datetime.datetime


class DataQARunOut(BaseModel):
    id: int
    name: str
    dataset_id: int
    run_id: str
    status: RunStatus
    approved: bool
    created_at: datetime.datetime
    run_args: typing.Optional[RunArgs]
