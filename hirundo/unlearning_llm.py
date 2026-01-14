import datetime
import json
import typing
from collections.abc import AsyncGenerator, Generator
from enum import Enum
from typing import Literal, overload

import httpx
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.dataset_qa import HirundoError
from hirundo.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 200  # Max 200 retries for HTTP SSE connection


class ModelSourceType(str, Enum):
    HUGGINGFACE_TRANSFORMERS = "huggingface_transformers"
    LOCAL_TRANSFORMERS = "local_transformers"


class HuggingFaceTransformersModel(BaseModel):
    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    revision: str | None = None
    code_revision: str | None = None
    model_name: str
    token: str | None = None


class HuggingFaceTransformersModelOutput(BaseModel):
    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    model_name: str


class LocalTransformersModel(BaseModel):
    type: Literal[ModelSourceType.LOCAL_TRANSFORMERS] = (
        ModelSourceType.LOCAL_TRANSFORMERS
    )
    revision: None = None
    code_revision: None = None
    local_path: str


LlmSources = HuggingFaceTransformersModel | LocalTransformersModel
LlmSourcesOutput = HuggingFaceTransformersModelOutput | LocalTransformersModel


class LlmModel(BaseModel):
    id: int | None = None
    organization_id: int | None = None
    model_name: str
    model_source: LlmSources
    archive_existing_runs: bool = True

    def create(
        self,
        replace_if_exists: bool = False,
    ) -> int:
        llm_model_response = requests.post(
            f"{API_HOST}/unlearning-llm/llm/",
            json={
                **self.model_dump(mode="json"),
                "replace_if_exists": replace_if_exists,
            },
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        llm_model_id = llm_model_response.json()["id"]
        self.id = llm_model_id
        return llm_model_id

    @staticmethod
    def get_by_id(llm_model_id: int) -> "LlmModelOut":
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/{llm_model_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        return LlmModelOut.from_response(llm_model_response.json())

    @staticmethod
    def get_by_name(llm_model_name: str) -> "LlmModelOut":
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/by-name/{llm_model_name}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        return LlmModelOut.from_response(llm_model_response.json())

    @staticmethod
    def list(organization_id: int | None = None) -> list["LlmModelOut"]:
        params = {}
        if organization_id is not None:
            params["model_organization_id"] = organization_id
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/",
            params=params,
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        llm_model_json = llm_model_response.json()
        return [LlmModelOut.from_response(llm_model) for llm_model in llm_model_json]

    @staticmethod
    def delete_by_id(llm_model_id: int) -> None:
        llm_model_response = requests.delete(
            f"{API_HOST}/unlearning-llm/llm/{llm_model_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        logger.info("Deleted LLM model with ID: %s", llm_model_id)

    def delete(self) -> None:
        if not self.id:
            raise ValueError("No LLM model has been created")
        self.delete_by_id(self.id)

    def update(
        self,
        model_name: str | None = None,
        model_source: LlmSources | None = None,
        archive_existing_runs: bool | None = None,
    ) -> None:
        if not self.id:
            raise ValueError("No LLM model has been created")
        payload: dict[str, typing.Any] = {
            "model_name": model_name,
            "model_source": model_source.model_dump(mode="json")
            if model_source
            else None,
            "archive_existing_runs": archive_existing_runs,
            "organization_id": self.organization_id,
        }
        llm_model_response = requests.put(
            f"{API_HOST}/unlearning-llm/llm/{self.id}",
            json=payload,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        if model_name is not None:
            self.model_name = model_name
        if model_source is not None:
            self.model_source = model_source
        if archive_existing_runs is not None:
            self.archive_existing_runs = archive_existing_runs


class LlmModelOut(BaseModel):
    id: int
    organization_id: int
    creator_id: int
    creator_name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_name: str
    model_source: LlmSourcesOutput

    @staticmethod
    def from_response(response_payload: dict[str, typing.Any]) -> "LlmModelOut":
        return LlmModelOut(
            id=response_payload["id"],
            organization_id=response_payload["organization_id"],
            creator_id=response_payload["creator_id"],
            creator_name=response_payload["creator_name"],
            created_at=response_payload["created_at"],
            updated_at=response_payload["updated_at"],
            model_name=response_payload["model_name"],
            model_source=response_payload["model_source"],
        )


class DatasetType(str, Enum):
    NORMAL = "normal"
    BIAS = "bias"
    UNBIAS = "unbias"


class UnlearningLlmAdvancedOptions(BaseModel):
    max_tokens_for_model: dict[DatasetType, int] | int | None = None


class BiasType(str, Enum):
    ALL = "ALL"
    RACE = "RACE"
    NATIONALITY = "NATIONALITY"
    GENDER = "GENDER"
    PHYSICAL_APPEARANCE = "PHYSICAL_APPEARANCE"
    RELIGION = "RELIGION"
    AGE = "AGE"


class UtilityType(str, Enum):
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"


class DefaultUtility(BaseModel):
    utility_type: Literal[UtilityType.DEFAULT] = UtilityType.DEFAULT


class HirundoCSVDataset(BaseModel):
    type: Literal["HirundoCSV"] = "HirundoCSV"
    csv_url: str


class HuggingFaceDataset(BaseModel):
    type: Literal["HuggingFaceDataset"] = "HuggingFaceDataset"
    hugging_face_dataset_name: str


CustomDataset = HirundoCSVDataset | HuggingFaceDataset


class CustomUtility(BaseModel):
    utility_type: Literal[UtilityType.CUSTOM] = UtilityType.CUSTOM
    dataset: CustomDataset


class BiasBehavior(BaseModel):
    type: Literal["BIAS"] = "BIAS"
    bias_type: BiasType


class HallucinationType(str, Enum):
    GENERAL = "GENERAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    DEFENSE = "DEFENSE"


class HallucinationBehavior(BaseModel):
    type: Literal["HALLUCINATION"] = "HALLUCINATION"
    hallucination_type: HallucinationType


class SecurityBehavior(BaseModel):
    type: Literal["SECURITY"] = "SECURITY"


class CustomBehavior(BaseModel):
    type: Literal["CUSTOM"] = "CUSTOM"
    biased_dataset: CustomDataset
    unbiased_dataset: CustomDataset


TargetBehavior = (
    BiasBehavior | HallucinationBehavior | SecurityBehavior | CustomBehavior
)

TargetUtility = DefaultUtility | CustomUtility


class LlmRunInfo(BaseModel):
    organization_id: int | None = None
    name: str | None = None
    model_id: int | None = None
    target_behaviors: list[TargetBehavior]
    target_utilities: list[TargetUtility]
    advanced_options: UnlearningLlmAdvancedOptions | None = None


class BiasRunInfo(BaseModel):
    bias_type: BiasType
    organization_id: int | None = None
    name: str | None = None
    target_utilities: list[TargetUtility] | None = None
    advanced_options: UnlearningLlmAdvancedOptions | None = None

    def to_run_info(self) -> LlmRunInfo:
        default_utilities: list[TargetUtility] = (
            [DefaultUtility()]
            if self.target_utilities is None
            else list(self.target_utilities)
        )
        return LlmRunInfo(
            organization_id=self.organization_id,
            name=self.name,
            target_behaviors=[BiasBehavior(bias_type=self.bias_type)],
            target_utilities=default_utilities,
            advanced_options=self.advanced_options,
        )


OutputLlm = dict[str, object]
BehaviorOptions = TargetBehavior
UtilityOptions = TargetUtility
CeleryTaskState = str


class OutputUnlearningLlmRun(BaseModel):
    id: int
    name: str
    model_id: int
    model: OutputLlm
    target_behaviors: list[BehaviorOptions]
    target_utilities: list[UtilityOptions]
    advanced_options: UnlearningLlmAdvancedOptions | None
    run_id: str
    mlflow_run_id: str | None
    status: CeleryTaskState
    approved: bool
    created_at: datetime.datetime
    completed_at: datetime.datetime | None
    pre_process_progress: float
    optimization_progress: float
    post_process_progress: float


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
    RunStatus.STARTED.value: "LLM unlearning run in progress",
    RunStatus.PENDING.value: "LLM unlearning run queued and not yet started",
    RunStatus.SUCCESS.value: "LLM unlearning run completed successfully",
    RunStatus.FAILURE.value: "LLM unlearning run failed",
    RunStatus.AWAITING_MANUAL_APPROVAL.value: "Awaiting manual approval",
    RunStatus.RETRY.value: "LLM unlearning run failed. Retrying",
    RunStatus.REVOKED.value: "LLM unlearning run was cancelled",
    RunStatus.REJECTED.value: "LLM unlearning run was rejected",
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


class LlmUnlearningRun:
    @staticmethod
    def launch(model_id: int, run_info: LlmRunInfo | BiasRunInfo) -> str:
        resolved_run_info = (
            run_info.to_run_info() if isinstance(run_info, BiasRunInfo) else run_info
        )
        run_response = requests.post(
            f"{API_HOST}/unlearning-llm/run/{model_id}",
            json=resolved_run_info.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        run_response_json = run_response.json() if run_response.content else {}
        if isinstance(run_response_json, str):
            return run_response_json
        run_id = (
            run_response_json.get("run_id")
            or run_response_json.get("hir_run_id")
            or run_response_json.get("id")
        )
        if not run_id:
            raise ValueError("No run ID returned from launch request")
        return run_id

    @staticmethod
    def cancel(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/cancel/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def rename(run_id: str, new_name: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/rename/{run_id}",
            json={"new_name": new_name},
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def archive(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/archive/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def restore(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/restore/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def list(
        organization_id: int | None = None,
        archived: bool = False,
    ) -> list[OutputUnlearningLlmRun]:
        params: dict[str, bool | int] = {"archived": archived}
        if organization_id is not None:
            params["unlearning_organization_id"] = organization_id
        run_response = requests.get(
            f"{API_HOST}/unlearning-llm/run/list",
            params=params,
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        response_json = run_response.json()
        if isinstance(response_json, list):
            return [
                OutputUnlearningLlmRun.model_validate(run_payload)
                for run_payload in response_json
            ]
        return [OutputUnlearningLlmRun.model_validate(response_json)]

    @staticmethod
    def _check_run_by_id(run_id: str, retry=0) -> Generator[dict, None, None]:
        if retry > MAX_RETRIES:
            raise HirundoError("Max retries reached")
        last_event = None
        with httpx.Client(timeout=httpx.Timeout(None, connect=5.0)) as client:
            for sse in iter_sse_retrying(
                client,
                "GET",
                f"{API_HOST}/unlearning-llm/run/{run_id}",
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
        last_state = None
        if last_event and "data" in last_event:
            last_state = last_event["data"].get("state") or last_event["data"].get(
                "status"
            )
        if not last_event or last_state == RunStatus.PENDING.value:
            LlmUnlearningRun._check_run_by_id(run_id, retry + 1)

    @staticmethod
    def _handle_failure(iteration: dict) -> None:
        if iteration.get("result"):
            raise HirundoError(
                f"LLM unlearning run failed with error: {iteration['result']}"
            )
        raise HirundoError("LLM unlearning run failed with an unknown error")

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: Literal[True]
    ) -> typing.Any | None: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: Literal[False] = False
    ) -> typing.Any: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool
    ) -> typing.Any | None: ...

    @staticmethod
    def check_run_by_id(run_id: str, stop_on_manual_approval: bool = False):
        """
        Check the status of a run given its ID

        Args:
            run_id: The `run_id` produced by a `launch` call
            stop_on_manual_approval: If True, the function will return `None` if the run is awaiting manual approval

        Returns:
            The result payload for the run, if available

        Raises:
            HirundoError: If the maximum number of retries is reached or if the run fails
        """
        logger.debug("Checking run with ID: %s", run_id)
        with logging_redirect_tqdm():
            t = tqdm(total=100.0)
            for iteration in LlmUnlearningRun._check_run_by_id(run_id):
                state = iteration.get("state") or iteration.get("status")
                if state in STATUS_TO_PROGRESS_MAP:
                    t.set_description(STATUS_TO_TEXT_MAP[state])
                    t.n = STATUS_TO_PROGRESS_MAP[state]
                    logger.debug("Setting progress to %s", t.n)
                    t.refresh()
                    if state in [
                        RunStatus.FAILURE.value,
                        RunStatus.REJECTED.value,
                        RunStatus.REVOKED.value,
                    ]:
                        logger.error(
                            "State is failure, rejected, or revoked: %s",
                            state,
                        )
                        LlmUnlearningRun._handle_failure(iteration)
                    elif state == RunStatus.SUCCESS.value:
                        t.close()
                        return iteration.get("result") or iteration
                    elif (
                        state == RunStatus.AWAITING_MANUAL_APPROVAL.value
                        and stop_on_manual_approval
                    ):
                        t.close()
                        return None
                elif state is None:
                    if (
                        iteration.get("result")
                        and isinstance(iteration["result"], dict)
                        and iteration["result"].get("result")
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
                            current_progress_percentage = t.n
                        else:
                            stage = "Unknown progress state"
                            current_progress_percentage = t.n
                        desc = (
                            "LLM unlearning run completed. Uploading results"
                            if current_progress_percentage == 100.0
                            else stage
                        )
                        t.set_description(desc)
                        t.n = current_progress_percentage
                        logger.debug("Setting progress to %s", t.n)
                        t.refresh()
        raise HirundoError("LLM unlearning run failed with an unknown error")

    @staticmethod
    def check_run(run_id: str, stop_on_manual_approval: bool = False):
        """
        Check the status of the given run.

        Returns:
            The result payload for the run, if available
        """
        return LlmUnlearningRun.check_run_by_id(run_id, stop_on_manual_approval)

    @staticmethod
    async def acheck_run_by_id(run_id: str, retry=0) -> AsyncGenerator[dict, None]:
        """
        Async version of :func:`check_run_by_id`

        Check the status of a run given its ID.

        This generator will produce values to show progress of the run.

        Args:
            run_id: The `run_id` produced by a `launch` call
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
                f"{API_HOST}/unlearning-llm/run/{run_id}",
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
        last_state = None
        if last_event and "data" in last_event:
            last_state = last_event["data"].get("state") or last_event["data"].get(
                "status"
            )
        if not last_event or last_state == RunStatus.PENDING.value:
            LlmUnlearningRun.acheck_run_by_id(run_id, retry + 1)

    @staticmethod
    async def acheck_run(run_id: str) -> AsyncGenerator[dict, None]:
        """
        Async version of :func:`check_run`

        Check the status of the given run.

        This generator will produce values to show progress of the run.

        Yields:
            Each event will be a dict, where:
            - `"state"` is PENDING, STARTED, RETRY, FAILURE or SUCCESS
            - `"result"` is a string describing the progress as a percentage for a PENDING state, or the error for a FAILURE state or the results for a SUCCESS state

        """
        async for iteration in LlmUnlearningRun.acheck_run_by_id(run_id):
            yield iteration
