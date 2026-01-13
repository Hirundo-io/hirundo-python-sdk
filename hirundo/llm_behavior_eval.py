import datetime
import json
import typing
from collections.abc import AsyncGenerator, Generator
from enum import Enum

import httpx
from pydantic import BaseModel, ConfigDict

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.logger import get_logger

logger = get_logger(__name__)


class LlmBehaviorEvalError(Exception):
    """
    Custom exception used to indicate errors in `hirundo` LLM behavior eval runs.
    """

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


class BiasType(str, Enum):
    ALL = "ALL"
    RACE = "RACE"
    NATIONALITY = "NATIONALITY"
    GENDER = "GENDER"
    PHYSICAL_APPEARANCE = "PHYSICAL_APPEARANCE"
    RELIGION = "RELIGION"
    AGE = "AGE"


class JudgeModel(BaseModel):
    path_or_repo_id: str
    token: typing.Optional[str] = None
    batch_size: typing.Optional[int] = None
    output_tokens: typing.Optional[int] = None
    use_4bit: typing.Optional[bool] = None


class EvalRunInfo(BaseModel):
    organization_id: typing.Optional[int] = None
    name: typing.Optional[str] = None
    model_id: typing.Optional[int] = None
    source_run_id: typing.Optional[str] = None
    file_path: typing.Optional[str] = None
    preset_type: typing.Optional[PresetType] = None
    bias_type: typing.Optional[BiasType] = None
    judge_model: typing.Optional[JudgeModel] = None


class OutputLlm(BaseModel):
    model_config = {"extra": "allow"}

    id: int
    organization_id: int
    creator_id: int
    creator_name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_name: str
    model_source: dict


class OutputUnlearningLlmRun(BaseModel):
    model_config = {"extra": "allow"}

    id: int
    name: str
    run_id: str
    status: str
    created_at: datetime.datetime


class LlmEvalMetricRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    benchmark: str
    metric: str
    original: typing.Optional[typing.Union[float, str]] = None
    post_unlearning: typing.Optional[typing.Union[float, str]] = None
    reduction_percent: typing.Optional[float] = None
    subset: typing.Optional[str] = None


class LlmEvalMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    rows: list[LlmEvalMetricRow]


class EvalRunRecord(BaseModel):
    id: int
    name: str
    model_id: typing.Optional[int]
    model: typing.Optional[OutputLlm]
    source_run_id: typing.Optional[str]
    source_run: typing.Optional[OutputUnlearningLlmRun]
    file_path: typing.Optional[str]
    preset_type: typing.Optional[PresetType]
    bias_type: typing.Optional[BiasType]
    judge_model: typing.Optional[JudgeModel]
    run_id: str
    mlflow_run_id: typing.Optional[str]
    status: str
    created_at: datetime.datetime
    pre_process_progress: float
    optimization_progress: float
    post_process_progress: float
    metrics: typing.Optional[LlmEvalMetrics] = None
    responses_zip_url: typing.Optional[str] = None


class LlmBehaviorEval:
    def __init__(self, run_id: typing.Optional[str] = None):
        self.run_id = run_id

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
        model_or_run: typing.Union[ModelOrRun, str],
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
            raise LlmBehaviorEvalError(
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
            raise ValueError("No run has been started")
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
            raise ValueError("No run has been started")
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
            raise ValueError("No run has been started")
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
            raise ValueError("No run has been started")
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
        organization_id: typing.Optional[int] = None,
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
    def stream_results_by_id(run_id: str) -> Generator[dict, None, None]:
        """
        Stream evaluation results for a run.
        """
        with httpx.Client(timeout=httpx.Timeout(None, connect=5.0)) as client:
            for sse_event in iter_sse_retrying(
                client,
                "GET",
                f"{API_HOST}/llm-behavior-eval/run/{run_id}",
                headers=get_headers(),
            ):
                if sse_event.event == "ping":
                    continue
                try:
                    yield json.loads(sse_event.data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE payload received: %s", sse_event.data)
                    yield {"data": sse_event.data}

    def stream_results(self) -> Generator[dict, None, None]:
        if not self.run_id:
            raise ValueError("No run has been started")
        yield from self.stream_results_by_id(self.run_id)

    @staticmethod
    async def astream_results_by_id(run_id: str) -> AsyncGenerator[dict, None]:
        """
        Async stream evaluation results for a run.
        """
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
                try:
                    yield json.loads(sse_event.data)
                except json.JSONDecodeError:
                    logger.debug(
                        "Non-JSON SSE payload received: %s", sse_event.data
                    )
                    yield {"data": sse_event.data}

    async def astream_results(self) -> AsyncGenerator[dict, None]:
        if not self.run_id:
            raise ValueError("No run has been started")
        async for payload in self.astream_results_by_id(self.run_id):
            yield payload
