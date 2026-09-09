import datetime
from collections.abc import AsyncGenerator, Generator
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, cast, overload

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hirundo._env import API_HOST
from hirundo._generated.wire_models import (
    CreateLlm,
    ServerUnlearningLlmModelsRunRunInfo,
)
from hirundo._generated.wire_models import (
    DatasetType as GeneratedDatasetType,
)
from hirundo._generated.wire_models import (
    OutputLlm as GeneratedOutputLlm,
)
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._llm_pipeline import get_hf_pipeline_for_run_given_model
from hirundo._llm_sources import LlmSources, LlmSourcesOutput
from hirundo._run_checking import (
    STATUS_TO_PROGRESS_MAP,
    aiter_run_events,
    build_status_text_map,
    get_state,
    handle_run_failure,
    iter_run_events,
    update_progress_from_result,
)
from hirundo._run_status import RunStatus
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.dataset_qa import HirundoError
from hirundo.llm_bias_type import BBQBiasType
from hirundo.logger import get_logger

if TYPE_CHECKING:
    from torch import device as torch_device
    from transformers.configuration_utils import PretrainedConfig
    from transformers.pipelines.base import Pipeline

logger = get_logger(__name__)

DatasetType = GeneratedDatasetType


class LlmModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    id: int | None = None
    organization_id: int | None = None
    model_name: str
    model_source: LlmSources
    archive_existing_runs: bool = True

    def create(
        self,
        replace_if_exists: bool = False,
    ) -> int:
        payload = {
            **self.model_dump(mode="json", exclude={"id"}),
            "replace_if_exists": replace_if_exists,
        }
        CreateLlm.model_validate(payload)
        llm_model_response = requests.post(
            f"{API_HOST}/unlearning-llm/llm/",
            json=payload,
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
        response_payload = llm_model_response.json()
        GeneratedOutputLlm.model_validate(response_payload)
        return LlmModelOut.model_validate(response_payload)

    @staticmethod
    def get_by_name(llm_model_name: str) -> "LlmModelOut":
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/by-name/{llm_model_name}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        response_payload = llm_model_response.json()
        GeneratedOutputLlm.model_validate(response_payload)
        return LlmModelOut.model_validate(response_payload)

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
        return [
            LlmModelOut.model_validate(
                GeneratedOutputLlm.model_validate(llm_model).model_dump(mode="json")
            )
            for llm_model in llm_model_json
        ]

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
        payload: dict[str, JsonValue] = {
            "model_name": model_name,
            "model_source": cast(
                "dict[str, JsonValue]", model_source.model_dump(mode="json")
            )
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

    def get_hf_pipeline_for_run(
        self,
        run_id: str,
        config: "PretrainedConfig | None" = None,
        device: "str | int | torch_device | None" = None,
        device_map: str | dict[str, int | str] | None = None,
        trust_remote_code: bool = False,
    ) -> "Pipeline":
        return get_hf_pipeline_for_run_given_model(
            self, run_id, config, device, device_map, trust_remote_code
        )


class LlmModelOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    id: int
    organization_id: int
    creator_id: int
    creator_name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_name: str
    model_source: LlmSourcesOutput

    def get_hf_pipeline_for_run(
        self,
        run_id: str,
        config: "PretrainedConfig | None" = None,
        device: "str | int | torch_device | None" = None,
        device_map: str | dict[str, int | str] | None = None,
        trust_remote_code: bool = False,
        token: str | None = None,
    ) -> "Pipeline":
        return get_hf_pipeline_for_run_given_model(
            self,
            run_id,
            config,
            device,
            device_map,
            trust_remote_code,
            token=token,
        )


class UnlearningLlmAdvancedOptions(BaseModel):
    """Advanced launch options matching the API request fields."""

    max_tokens_for_model: dict[DatasetType, int] | int | None = None


class HirundoCSVDataset(BaseModel):
    """Public Hirundo CSV dataset matching the API request fields."""

    csv_url: str
    type: Literal["HirundoCSV"] = "HirundoCSV"


class HuggingFaceDataset(BaseModel):
    """Public Hugging Face dataset matching the API request fields."""

    hugging_face_dataset_name: str
    token: str | None = None
    token_id: int | None = None
    type: Literal["HuggingFaceDataset"] = "HuggingFaceDataset"


CustomDataset = HirundoCSVDataset | HuggingFaceDataset


class CustomUtility(BaseModel):
    dataset: CustomDataset


class BiasBehavior(BaseModel):
    """Bias behavior with the SDK's backend-only bias type omitted."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["BIAS"] = "BIAS"


class HallucinationType(str, Enum):
    GENERAL = "GENERAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    DEFENSE = "DEFENSE"


class HallucinationBehavior(BaseModel):
    """Public hallucination behavior backed by generated API fields."""

    type: Literal["HALLUCINATION"] = "HALLUCINATION"
    hallucination_type: HallucinationType


class SecurityBehavior(BaseModel):
    """Public security behavior matching the API request fields."""

    type: Literal["SECURITY"] = "SECURITY"


class CustomBehavior(BaseModel):
    type: Literal["CUSTOM"] = "CUSTOM"
    biased_dataset: CustomDataset
    unbiased_dataset: CustomDataset


TargetBehavior = Annotated[
    BiasBehavior | HallucinationBehavior | SecurityBehavior | CustomBehavior,
    Field(discriminator="type"),
]


class OutputBiasBehavior(BaseModel):
    type: Literal["BIAS"] = "BIAS"
    bias_type: BBQBiasType


OutputBehaviorOptions = (
    OutputBiasBehavior | HallucinationBehavior | SecurityBehavior | CustomBehavior
)


class LlmRunInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    organization_id: int | None = None
    name: str | None = None
    target_behaviors: list[TargetBehavior]
    target_utilities: list[CustomUtility] = Field(default_factory=list)
    advanced_options: UnlearningLlmAdvancedOptions | None = None


OutputLlm = dict[str, JsonValue]
CeleryTaskState = str


class OutputUnlearningLlmRun(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    id: int
    name: str
    model_id: int
    model: OutputLlm
    target_behaviors: list[OutputBehaviorOptions]
    target_utilities: list[CustomUtility]
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

    deleted_at: datetime.datetime | None = None


STATUS_TO_TEXT_MAP = build_status_text_map("LLM unlearning")


class LlmUnlearningRun:
    @staticmethod
    def _build_launch_payload(run_info: LlmRunInfo) -> dict[str, JsonValue]:
        """
        Build the JSON payload for an LLM unlearning launch request.

        Args:
            run_info: The `LlmRunInfo` request model to serialize.

        Returns:
            A JSON-serializable payload derived from
            `run_info.model_dump(mode="json")`. Bias targets include the
            backend-only `bias_type` field set to `BBQBiasType.ALL.value`.
        """
        payload = cast("dict[str, JsonValue]", run_info.model_dump(mode="json"))
        target_behaviors = payload["target_behaviors"]
        if not isinstance(target_behaviors, list):
            raise TypeError("target_behaviors must serialize as a list")
        for target_behavior in target_behaviors:
            if not isinstance(target_behavior, dict):
                raise TypeError("each target behavior must serialize as an object")
            if target_behavior["type"] == "BIAS":
                target_behavior["bias_type"] = BBQBiasType.ALL.value
        ServerUnlearningLlmModelsRunRunInfo.model_validate(payload)
        return payload

    @staticmethod
    def launch(model_id: int, run_info: LlmRunInfo) -> str:
        run_response = requests.post(
            f"{API_HOST}/unlearning-llm/run/{model_id}",
            json=LlmUnlearningRun._build_launch_payload(run_info),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        run_response_json = run_response.json() if run_response.content else {}
        if isinstance(run_response_json, str):
            return run_response_json
        run_id = run_response_json.get("run_id")
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
    def _check_run_by_id(
        run_id: str, retry: int = 0
    ) -> Generator[dict[str, JsonValue], None, None]:
        yield from iter_run_events(
            f"{API_HOST}/unlearning-llm/run/{run_id}",
            headers=get_headers(),
            retry=retry,
            status_keys=("state", "status"),
            error_cls=HirundoError,
            log=logger,
        )

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: Literal[True]
    ) -> JsonValue | None: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: Literal[False] = False
    ) -> JsonValue: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool
    ) -> JsonValue | None: ...

    @staticmethod
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool = False
    ) -> JsonValue | None:
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
                state = get_state(iteration, ("state", "status"))
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
                        t.close()
                        handle_run_failure(
                            iteration,
                            error_cls=HirundoError,
                            run_label="LLM unlearning",
                        )
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
                    update_progress_from_result(
                        iteration,
                        t,
                        uploading_text="LLM unlearning run completed. Uploading results",
                        log=logger,
                    )
        raise HirundoError("LLM unlearning run failed with an unknown error")

    @staticmethod
    def check_run(
        run_id: str, stop_on_manual_approval: bool = False
    ) -> JsonValue | None:
        """
        Check the status of the given run.

        Args:
            run_id: Identifier of the unlearning run to monitor.
            stop_on_manual_approval: Return while the run awaits manual approval.

        Returns:
            The result payload for the run, if available
        """
        return LlmUnlearningRun.check_run_by_id(run_id, stop_on_manual_approval)

    @staticmethod
    async def acheck_run_by_id(
        run_id: str, retry: int = 0
    ) -> AsyncGenerator[dict[str, JsonValue], None]:
        """
        Async version of :func:`check_run_by_id`

        Check the status of a run given its ID.

        This generator will produce values to show progress of the run.

        Note: This function does not handle errors nor show progress. It is expected that you do that.

        Args:
            run_id: The `run_id` produced by a `launch` call
            retry: A number used to track the number of retries to limit re-checks. *Do not* provide this value manually.

        Returns:
            An asynchronous generator of run-status event dictionaries.

        Yields:
            Each event will be a dict, where:
            - `"state"` is PENDING, STARTED, RETRY, FAILURE or SUCCESS
            - `"result"` is a string describing the progress as a percentage for a PENDING state, or the error for a FAILURE state or the results for a SUCCESS state

        """
        logger.debug("Checking run with ID: %s", run_id)
        async for iteration in aiter_run_events(
            f"{API_HOST}/unlearning-llm/run/{run_id}",
            headers=get_headers(),
            retry=retry,
            status_keys=("state", "status"),
            error_cls=HirundoError,
            log=logger,
        ):
            yield iteration

    @staticmethod
    async def acheck_run(run_id: str) -> AsyncGenerator[dict[str, JsonValue], None]:
        """
        Async version of :func:`check_run`

        Check the status of the given run.

        This generator will produce values to show progress of the run.

        Args:
            run_id: Identifier of the unlearning run to monitor.

        Returns:
            An asynchronous generator of run-status event dictionaries.

        Yields:
            Each event will be a dict, where:
            - `"state"` is PENDING, STARTED, RETRY, FAILURE or SUCCESS
            - `"result"` is a string describing the progress as a percentage for a PENDING state, or the error for a FAILURE state or the results for a SUCCESS state

        """
        async for iteration in LlmUnlearningRun.acheck_run_by_id(run_id):
            yield iteration
