import datetime
import typing
from collections.abc import AsyncGenerator, Generator
from enum import Enum
from typing import overload

import httpx
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._hirundo_error import HirundoError
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from hirundo._llm_sources import HuggingFaceTransformersModelOutput, LlmSourcesOutput
from hirundo._model_access import (
    validate_huggingface_model_access,
    validate_judge_model_access,
)
from hirundo._run_checking import (
    DEFAULT_MAX_RETRIES,
    STATUS_TO_PROGRESS_MAP,
    build_status_text_map,
    get_state,
    handle_run_failure,
    update_progress_from_result,
)
from hirundo._run_status import RunStatus
from hirundo._sse_event_data import SseRunEventData, _parse_sse_payload
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.llm_behavior_eval_results import LlmBehaviorEvalResults
from hirundo.llm_bias_type import BBQBiasType, UnqoverBiasType
from hirundo.logger import get_logger
from hirundo.unlearning_llm import LlmModel
from hirundo.unzip import download_and_extract_llm_behavior_eval_zip

logger = get_logger(__name__)


STATUS_TO_TEXT_MAP = build_status_text_map("LLM behavior eval")


class HirundoLlmBehaviorEvalError(HirundoError):
    pass


class ModelOrRun(str, Enum):
    MODEL = "model"
    RUN = "run"


class PresetType(str, Enum):
    BBQ_BIAS = "BBQ_BIAS"
    BBQ_UNBIAS = "BBQ_UNBIAS"
    UNQOVER_BIAS = "UNQOVER_BIAS"
    HALU_EVAL = "HALU_EVAL"
    MED_HALLU = "MED_HALLU"
    INJECTION_EVAL = "INJECTION_EVAL"


class JudgeModel(BaseModel):
    path_or_repo_id: str
    token: str | None = None
    batch_size: int | None = None
    output_tokens: int | None = None
    use_4bit: bool | None = None


class EvalRunInfo(BaseModel):
    organization_id: int | None = None
    name: str | None = None
    model_id: int | None = None
    source_run_id: str | None = None
    file_path: str | None = None
    preset_type: PresetType | None = None
    bias_type: BBQBiasType | UnqoverBiasType | None = None
    judge_model: JudgeModel | None = None


class OutputLlm(BaseModel):
    model_config = {"extra": "allow"}

    id: int
    organization_id: int
    creator_id: int
    creator_name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_name: str
    model_source: LlmSourcesOutput


class OutputUnlearningLlmRun(BaseModel):
    model_config = {"extra": "allow"}

    id: int
    name: str
    run_id: str
    model: OutputLlm | None = None
    status: str
    created_at: datetime.datetime


class LlmEvalMetricRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    benchmark: str
    metric: str
    original: float | str | None = None
    post_unlearning: float | str | None = None
    reduction_percent: float | None = None
    subset: str | None = None


class LlmEvalMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    rows: list[LlmEvalMetricRow]


class EvalRunRecord(BaseModel):
    id: int
    name: str
    model_id: int | None
    model: OutputLlm | None
    source_run_id: str | None
    source_run: OutputUnlearningLlmRun | None
    file_path: str | None
    preset_type: PresetType | None
    bias_type: BBQBiasType | UnqoverBiasType | None
    judge_model: JudgeModel | None
    run_id: str
    mlflow_run_id: str | None
    status: str
    created_at: datetime.datetime
    pre_process_progress: float
    optimization_progress: float
    post_process_progress: float
    metrics: LlmEvalMetrics | None = None
    responses_zip_url: str | None = None

    deleted_at: datetime.datetime | None = None


class LlmBehaviorEval:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id

    @staticmethod
    def _validate_model_access(model_or_run: ModelOrRun, run_info: EvalRunInfo) -> None:
        if run_info.judge_model is not None:
            validate_judge_model_access(
                path_or_repo_id=run_info.judge_model.path_or_repo_id,
                token=run_info.judge_model.token,
            )
        if model_or_run == ModelOrRun.MODEL and run_info.model_id is not None:
            llm_model = LlmModel.get_by_id(run_info.model_id)
            if isinstance(llm_model.model_source, HuggingFaceTransformersModelOutput):
                validate_huggingface_model_access(
                    model_name=llm_model.model_source.model_name,
                    token=llm_model.model_source.token,
                    model_role="LLM",
                )

    @staticmethod
    def _parse_eval_run_record(response_payload: dict) -> EvalRunRecord:
        model_payload = response_payload.get("model")
        source_run_payload = response_payload.get("source_run")
        judge_model_payload = response_payload.get("judge_model")
        metrics_payload = response_payload.get("metrics")

        model = (
            OutputLlm.model_validate(model_payload)
            if isinstance(model_payload, dict)
            else None
        )
        source_run = (
            OutputUnlearningLlmRun.model_validate(source_run_payload)
            if isinstance(source_run_payload, dict)
            else None
        )
        judge_model = (
            JudgeModel.model_validate(judge_model_payload)
            if isinstance(judge_model_payload, dict)
            else None
        )
        if isinstance(metrics_payload, dict):
            metrics = LlmEvalMetrics.model_validate(metrics_payload)
        elif isinstance(metrics_payload, list):
            metric_rows = [
                LlmEvalMetricRow.model_validate(metric_row)
                for metric_row in metrics_payload
                if isinstance(metric_row, dict)
            ]
            metrics = LlmEvalMetrics(rows=metric_rows)
        else:
            metrics = None

        return EvalRunRecord(
            id=response_payload["id"],
            name=response_payload["name"],
            model_id=response_payload.get("model_id"),
            model=model,
            source_run_id=response_payload.get("source_run_id"),
            source_run=source_run,
            file_path=response_payload.get("file_path"),
            preset_type=response_payload.get("preset_type"),
            bias_type=response_payload.get("bias_type"),
            judge_model=judge_model,
            run_id=response_payload["run_id"],
            mlflow_run_id=response_payload.get("mlflow_run_id"),
            status=response_payload["status"],
            created_at=response_payload["created_at"],
            pre_process_progress=response_payload.get("pre_process_progress", 0.0),
            optimization_progress=response_payload.get("optimization_progress", 0.0),
            post_process_progress=response_payload.get("post_process_progress", 0.0),
            metrics=metrics,
            responses_zip_url=response_payload.get("responses_zip_url"),
        )

    @staticmethod
    def launch_eval_run(
        model_or_run: ModelOrRun | str,
        run_info: EvalRunInfo,
    ) -> str:
        """
        Launch an LLM behavior evaluation run.

        Args:
            model_or_run: Whether the evaluation is based on a model or a run.
            run_info: The evaluation run parameters.

        Returns:
            The ID of the created evaluation run.
        """
        if isinstance(model_or_run, str):
            model_or_run_value = ModelOrRun(model_or_run)
        else:
            model_or_run_value = model_or_run

        LlmBehaviorEval._validate_model_access(model_or_run_value, run_info)

        response = requests.post(
            f"{API_HOST}/llm-behavior-eval/run/{model_or_run_value.value}",
            json=run_info.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_payload = response.json()
        run_identifier = (
            response_payload.get("run_id")
            or response_payload.get("hir_run_id")
            or response_payload.get("id")
        )
        if not run_identifier:
            raise HirundoLlmBehaviorEvalError(
                "Unable to determine the run ID from the response payload."
            )
        return run_identifier

    @staticmethod
    def cancel_by_id(run_id: str) -> None:
        """
        Cancel a running evaluation.
        """
        response = requests.patch(
            f"{API_HOST}/llm-behavior-eval/run/cancel/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def cancel(self) -> None:
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        self.cancel_by_id(self.run_id)

    @staticmethod
    def rename_by_id(run_id: str, new_name: str) -> None:
        """
        Rename an evaluation run.
        """
        response = requests.patch(
            f"{API_HOST}/llm-behavior-eval/run/rename/{run_id}",
            json={"new_name": new_name},
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def rename(self, new_name: str) -> None:
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        self.rename_by_id(self.run_id, new_name)

    @staticmethod
    def archive_by_id(run_id: str) -> None:
        """
        Archive an evaluation run.
        """
        response = requests.patch(
            f"{API_HOST}/llm-behavior-eval/run/archive/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def archive(self) -> None:
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        self.archive_by_id(self.run_id)

    @staticmethod
    def restore_by_id(run_id: str) -> None:
        """
        Restore an archived evaluation run.
        """
        response = requests.patch(
            f"{API_HOST}/llm-behavior-eval/run/restore/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def restore(self) -> None:
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        self.restore_by_id(self.run_id)

    @staticmethod
    def get_run_info_by_id(run_id: str) -> EvalRunRecord:
        """
        Retrieve the metadata for an evaluation run.
        """
        response = requests.get(
            f"{API_HOST}/llm-behavior-eval/run/info/{run_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_payload = response.json()
        return LlmBehaviorEval._parse_eval_run_record(response_payload)

    @staticmethod
    def list_runs(
        organization_id: int | None = None,
        archived: bool = False,
    ) -> list[EvalRunRecord]:
        """
        List evaluation runs.
        """
        response = requests.get(
            f"{API_HOST}/llm-behavior-eval/run/list",
            params={
                "eval_organization_id": organization_id,
                "archived": archived,
            },
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_payload = response.json()
        return [
            LlmBehaviorEval._parse_eval_run_record(run_record)
            for run_record in response_payload
        ]

    @staticmethod
    def _resolve_model_name(run_info: EvalRunRecord) -> str | None:
        if run_info.model and isinstance(
            run_info.model.model_source, HuggingFaceTransformersModelOutput
        ):
            return run_info.model.model_source.model_name
        if (
            run_info.source_run
            and run_info.source_run.model
            and isinstance(
                run_info.source_run.model.model_source,
                HuggingFaceTransformersModelOutput,
            )
        ):
            return run_info.source_run.model.model_source.model_name
        return None

    @staticmethod
    def _check_run_by_id(
        run_id: str, *, max_retries: int = DEFAULT_MAX_RETRIES
    ) -> Generator[SseRunEventData, None, None]:
        retry_count = 0
        while True:
            if retry_count > max_retries:
                raise HirundoLlmBehaviorEvalError("Max retries reached")
            last_payload = None
            with httpx.Client(timeout=httpx.Timeout(None, connect=5.0)) as client:
                for sse_event in iter_sse_retrying(
                    client,
                    "GET",
                    f"{API_HOST}/llm-behavior-eval/run/{run_id}",
                    headers=get_headers(),
                ):
                    if sse_event.event == "ping":
                        continue
                    payload = _parse_sse_payload(sse_event.data)
                    last_payload = payload
                    yield payload
            last_state = get_state(last_payload, ("state",)) if last_payload else None
            if last_payload is None or last_state == RunStatus.PENDING.value:
                retry_count += 1
                continue
            return

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: typing.Literal[True]
    ) -> LlmBehaviorEvalResults | None: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: typing.Literal[False] = False
    ) -> LlmBehaviorEvalResults: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool
    ) -> LlmBehaviorEvalResults | None: ...

    @staticmethod
    def check_run_by_id(
        run_id: str, stop_on_manual_approval: bool = False
    ) -> LlmBehaviorEvalResults | None:
        """
        Check the status of a run given its ID

        Args:
            run_id: The `run_id` produced by a `launch_eval_run` call
            stop_on_manual_approval: If True, the function will return `None` if the run is awaiting manual approval

        Returns:
            An LlmBehaviorEvalResults object with the results of the evaluation run

        Raises:
            HirundoLlmBehaviorEvalError: If the maximum number of retries is reached or if the run fails
        """
        logger.debug("Checking run with ID: %s", run_id)
        with logging_redirect_tqdm():
            progress_bar = tqdm(total=100.0)
            for iteration in LlmBehaviorEval._check_run_by_id(run_id):
                state = get_state(iteration, ("state",))
                if state in STATUS_TO_PROGRESS_MAP:
                    progress_bar.set_description(STATUS_TO_TEXT_MAP[state])
                    progress_bar.n = STATUS_TO_PROGRESS_MAP[state]
                    logger.debug("Setting progress to %s", progress_bar.n)
                    progress_bar.refresh()
                    if state in [
                        RunStatus.FAILURE.value,
                        RunStatus.REJECTED.value,
                        RunStatus.REVOKED.value,
                    ]:
                        logger.error(
                            "State is failure, rejected, or revoked: %s",
                            state,
                        )
                        progress_bar.close()
                        handle_run_failure(
                            iteration,
                            error_cls=HirundoLlmBehaviorEvalError,
                            run_label="LLM behavior eval",
                        )
                    elif state == RunStatus.SUCCESS.value:
                        progress_bar.close()
                        zip_temporary_url = iteration.result
                        if not zip_temporary_url or not isinstance(
                            zip_temporary_url, str
                        ):
                            raise HirundoLlmBehaviorEvalError(
                                "LLM behavior eval run completed without a results URL."
                            )
                        run_info = LlmBehaviorEval.get_run_info_by_id(run_id)
                        model_name = LlmBehaviorEval._resolve_model_name(run_info)
                        return download_and_extract_llm_behavior_eval_zip(
                            run_id,
                            zip_temporary_url,
                            model_name,
                        )
                    elif (
                        state == RunStatus.AWAITING_MANUAL_APPROVAL.value
                        and stop_on_manual_approval
                    ):
                        progress_bar.close()
                        return None
                elif state is None:
                    update_progress_from_result(
                        iteration,
                        progress_bar,
                        uploading_text="LLM behavior eval run completed. Uploading results",
                        log=logger,
                    )
        raise HirundoLlmBehaviorEvalError(
            "LLM behavior eval run failed with an unknown error in check_run_by_id"
        )

    @overload
    def check_run(
        self, stop_on_manual_approval: typing.Literal[True]
    ) -> LlmBehaviorEvalResults | None: ...

    @overload
    def check_run(
        self, stop_on_manual_approval: typing.Literal[False] = False
    ) -> LlmBehaviorEvalResults: ...

    def check_run(
        self, stop_on_manual_approval: bool = False
    ) -> LlmBehaviorEvalResults | None:
        """
        Check the status of the current active instance's run.

        Returns:
            An LlmBehaviorEvalResults object with the results of the evaluation run
        """
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        return self.check_run_by_id(self.run_id, stop_on_manual_approval)

    @staticmethod
    async def acheck_run_by_id(run_id: str) -> AsyncGenerator[SseRunEventData, None]:
        """
        Async version of :func:`check_run_by_id`

        Check the status of a run given its ID.

        This generator will produce values to show progress of the run.
        """
        logger.debug("Checking run with ID: %s", run_id)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=5.0)
        ) as client:
            async_iterator = await aiter_sse_retrying(
                client,
                "GET",
                f"{API_HOST}/llm-behavior-eval/run/{run_id}",
                headers=get_headers(),
            )
            async for sse_event in async_iterator:
                if sse_event.event == "ping":
                    continue
                yield _parse_sse_payload(sse_event.data)

    async def acheck_run(self) -> AsyncGenerator[SseRunEventData, None]:
        """
        Async version of :func:`check_run`

        Check the status of the current active instance's run.

        This generator will produce values to show progress of the run.

        Note: This function does not handle errors nor show progress. It is expected that you do that.
        """
        if not self.run_id:
            raise HirundoLlmBehaviorEvalError("No run has been started")
        async for iteration in self.acheck_run_by_id(self.run_id):
            yield iteration
