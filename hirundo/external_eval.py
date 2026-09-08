"""Public client models and methods for server-owned Inspect evaluations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._hirundo_error import HirundoError
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.llm_behavior_eval import ModelOrRun


class HirundoExternalEvalError(HirundoError):
    """Raised when an external evaluation response does not contain a run ID."""


class ExternalEvalRunInfo(BaseModel):
    """Settings for launching a catalog-owned Inspect evaluation."""

    model_config = ConfigDict(extra="forbid")

    organization_id: int | None = None
    name: str | None = None
    model_id: int | None = None
    source_run_id: str | None = None
    task_ids: list[str] = Field(min_length=1)
    sample_limit: int | None = Field(default=None, gt=0)

    def model_post_init(self, __context: object) -> None:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique")


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
    ) -> str:
        """Launch an Inspect evaluation for a saved model or unlearning run."""
        model_or_run_value = ModelOrRun(model_or_run)
        response = requests.post(
            f"{API_HOST}/external-evals/run/{model_or_run_value.value}",
            json=run_info.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        run_identifier = response.json().get("run_id")
        if not run_identifier:
            raise HirundoExternalEvalError(
                "Unable to determine the run ID from the response payload."
            )
        return run_identifier
