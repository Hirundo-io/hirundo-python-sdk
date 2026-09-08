"""Public client models and methods for server-owned Inspect evaluations."""

from collections.abc import AsyncGenerator
from typing import Annotated, Literal, overload
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hirundo._env import API_HOST, EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS
from hirundo._headers import get_headers
from hirundo._hirundo_error import HirundoError
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._run_checking import DEFAULT_MAX_RETRIES, get_state, handle_run_failure
from hirundo._run_status import RunStatus
from hirundo._sse_event_data import SseRunEventData
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.llm_behavior_eval import (
    HirundoLlmBehaviorEvalError,
    LlmBehaviorEval,
    ModelOrRun,
)
from hirundo.llm_behavior_eval_results import ExternalEvalResults
from hirundo.unzip import download_external_eval_zip


class HirundoExternalEvalError(HirundoError):
    """Raised when an external evaluation response does not contain a run ID."""


CanonicalTaskReference = Annotated[
    str,
    StringConstraints(pattern=r"^inspect_evals/[A-Za-z0-9_]+$"),
]

NonEmptyRunId = Annotated[str, StringConstraints(min_length=1)]


class ExternalEvalRunInfo(BaseModel):
    """Settings for launching a catalog-owned Inspect evaluation."""

    model_config = ConfigDict(extra="forbid")

    organization_id: int | None = None
    name: str | None = None
    model_id: int | None = None
    source_run_id: str | None = None
    task_ids: list[CanonicalTaskReference] = Field(min_length=1)
    sample_limit: int | None = Field(default=None, gt=0)

    def model_post_init(self, __context: object) -> None:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique")

    def validate_source(self, model_or_run: ModelOrRun) -> None:
        """Ensure the selected endpoint has exactly its required source ID."""
        if model_or_run is ModelOrRun.MODEL:
            if self.model_id is None or self.source_run_id is not None:
                raise ValueError(
                    "model launches require model_id and must not include source_run_id"
                )
            return
        if self.source_run_id is None or self.model_id is not None:
            raise ValueError(
                "run launches require source_run_id and must not include model_id"
            )


class ExternalEvalCatalogSource(BaseModel):
    """Versions used to build the deployment's Inspect catalogue."""

    inspect_ai: str
    inspect_evals: str


class ExternalEvalCatalogTask(BaseModel):
    """One Inspect task available to launch on the deployment."""

    id: str
    name: str
    sample_count: int | None


class ExternalEvalCatalogBenchmark(BaseModel):
    """A benchmark and its deployable Inspect tasks."""

    id: str
    title: str
    description: str
    category: str
    task_version: str
    paper_url: str | None
    tasks: list[ExternalEvalCatalogTask]


class ExternalEvalCatalog(BaseModel):
    """Deployment-visible Inspect evaluation catalogue."""

    schema_version: Literal[1]
    source: ExternalEvalCatalogSource
    benchmarks: list[ExternalEvalCatalogBenchmark]


class ExternalEvalLaunchResponse(BaseModel):
    """Server confirmation returned after queuing an Inspect evaluation."""

    message: str
    run_id: NonEmptyRunId


class ExternalEval:
    """Launch and discover server-owned Inspect evaluations."""

    @staticmethod
    def get_catalog() -> ExternalEvalCatalog:
        """Return the Inspect tasks available on the current deployment."""
        response = requests.get(
            f"{API_HOST}/external-evals/catalog",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        return ExternalEvalCatalog.model_validate(response.json())

    @staticmethod
    def launch_eval_run(
        model_or_run: ModelOrRun | Literal["model", "run"] | str,
        run_info: ExternalEvalRunInfo,
    ) -> ExternalEvalLaunchResponse:
        """Launch an Inspect evaluation for a saved model or unlearning run."""
        model_or_run_value = ModelOrRun(model_or_run)
        run_info.validate_source(model_or_run_value)
        response = requests.post(
            f"{API_HOST}/external-evals/run/{model_or_run_value.value}",
            json=run_info.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        try:
            return ExternalEvalLaunchResponse.model_validate(response.json())
        except ValueError as error:
            raise HirundoExternalEvalError(
                "Unable to determine the run ID from the response payload."
            ) from error

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise HirundoExternalEvalError(
                "External evaluation run IDs must be non-empty filename segments."
            )

    @staticmethod
    def _validate_result_url(result_url: str) -> None:
        if not EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS:
            return
        try:
            parsed_url = urlparse(result_url)
        except ValueError as error:
            raise HirundoExternalEvalError(
                "External evaluation result URL is malformed."
            ) from error
        result_origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if result_origin not in EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS:
            raise HirundoExternalEvalError(
                "External evaluation result URL is not in "
                "HIRUNDO_EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS."
            )

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stop_on_manual_approval: Literal[True],
    ) -> ExternalEvalResults | None: ...

    @staticmethod
    @overload
    def check_run_by_id(
        run_id: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stop_on_manual_approval: Literal[False] = False,
    ) -> ExternalEvalResults: ...

    @staticmethod
    def check_run_by_id(
        run_id: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stop_on_manual_approval: bool = False,
    ) -> ExternalEvalResults | None:
        """Poll an Inspect evaluation and download its unparsed result archive."""
        ExternalEval._validate_run_id(run_id)
        try:
            for event in LlmBehaviorEval._check_run_by_id(
                run_id, max_retries=max_retries
            ):
                state = get_state(event, ("state",))
                if state in {
                    RunStatus.FAILURE.value,
                    RunStatus.REJECTED.value,
                    RunStatus.REVOKED.value,
                }:
                    handle_run_failure(
                        event,
                        error_cls=HirundoExternalEvalError,
                        run_label="external evaluation",
                    )
                if state == RunStatus.SUCCESS.value:
                    result_url = event.result
                    if not isinstance(result_url, str) or not result_url:
                        raise HirundoExternalEvalError(
                            "External evaluation completed without a results URL."
                        )
                    ExternalEval._validate_result_url(result_url)
                    return download_external_eval_zip(run_id, result_url)
                if state == RunStatus.AWAITING_MANUAL_APPROVAL.value:
                    if stop_on_manual_approval:
                        return None
                    raise HirundoExternalEvalError(
                        "External evaluation is awaiting manual approval."
                    )
        except HirundoLlmBehaviorEvalError as error:
            raise HirundoExternalEvalError(str(error)) from error
        raise HirundoExternalEvalError(
            "External evaluation did not reach a terminal state"
        )

    @staticmethod
    async def acheck_run_by_id(
        run_id: str, *, max_retries: int = DEFAULT_MAX_RETRIES
    ) -> AsyncGenerator[SseRunEventData, None]:
        """Yield status events for an external evaluation without blocking.

        This method does not download result archives or raise terminal run
        failures. Consumers handle events as they arrive, including terminal
        states, and can await multiple runs concurrently.
        """
        ExternalEval._validate_run_id(run_id)
        try:
            async for event in LlmBehaviorEval.acheck_run_by_id(
                run_id, max_retries=max_retries
            ):
                yield event
        except HirundoLlmBehaviorEvalError as error:
            raise HirundoExternalEvalError(str(error)) from error
