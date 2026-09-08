from pathlib import Path
from typing import Any

import pytest
from hirundo import (
    CanonicalTaskReference,
    ExternalEval,
    ExternalEvalResults,
    ExternalEvalRunInfo,
    HirundoExternalEvalError,
    ModelOrRun,
)
from hirundo._run_status import RunStatus
from hirundo._sse_event_data import SseRunEventData
from hirundo.llm_behavior_eval import HirundoLlmBehaviorEvalError
from requests import HTTPError


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def test_launch_external_eval_run_serializes_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        captured_request["url"] = args[0]
        captured_request["json"] = kwargs["json"]
        return _Response({"message": "Run launched", "run_id": "inspect-run-id"})

    monkeypatch.setattr("hirundo.external_eval.get_headers", lambda: {})
    monkeypatch.setattr("hirundo.external_eval.requests.post", fake_post)

    launch = ExternalEval.launch_eval_run(
        ModelOrRun.RUN,
        ExternalEvalRunInfo(
            name="Inspect AIME",
            source_run_id="unlearning-run-id",
            task_ids=["inspect_evals/aime25"],
        ),
    )

    assert launch.run_id == "inspect-run-id"
    assert launch.message == "Run launched"
    assert captured_request["url"].endswith("/external-evals/run/run")
    assert captured_request["json"] == {
        "organization_id": None,
        "name": "Inspect AIME",
        "model_id": None,
        "source_run_id": "unlearning-run-id",
        "task_ids": ["inspect_evals/aime25"],
        "sample_limit": None,
    }


def test_get_external_eval_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        assert args[0].endswith("/external-evals/catalog")
        return _Response(
            {
                "schema_version": 1,
                "source": {"inspect_ai": "0.3.261", "inspect_evals": "0.19.0"},
                "benchmarks": [
                    {
                        "id": "aime25",
                        "title": "AIME 2025",
                        "description": "Mathematics benchmark",
                        "category": "reasoning",
                        "task_version": "1-A",
                        "paper_url": None,
                        "tasks": [
                            {
                                "id": "inspect_evals/aime25",
                                "name": "AIME 2025",
                                "sample_count": 30,
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("hirundo.external_eval.get_headers", lambda: {})
    monkeypatch.setattr("hirundo.external_eval.requests.get", fake_get)

    catalog = ExternalEval.get_catalog()

    assert catalog.benchmarks[0].tasks[0].id == "inspect_evals/aime25"


@pytest.mark.parametrize(
    "request_name",
    [
        "get",
        "post",
    ],
)
def test_external_eval_http_errors_are_raised(
    monkeypatch: pytest.MonkeyPatch,
    request_name: str,
) -> None:
    class ErrorResponse:
        status_code = 500

        def json(self) -> dict[str, str]:
            return {"reason": "service unavailable"}

        def raise_for_status(self) -> None:
            raise HTTPError("service unavailable")

    monkeypatch.setattr("hirundo.external_eval.get_headers", lambda: {})
    monkeypatch.setattr(
        f"hirundo.external_eval.requests.{request_name}",
        lambda *args, **kwargs: ErrorResponse(),
    )

    with pytest.raises(HTTPError, match="service unavailable"):
        if request_name == "get":
            ExternalEval.get_catalog()
        else:
            ExternalEval.launch_eval_run(
                ModelOrRun.MODEL,
                ExternalEvalRunInfo(
                    model_id=123,
                    task_ids=["inspect_evals/aime25"],
                ),
            )


def test_external_eval_run_info_rejects_duplicate_tasks() -> None:
    with pytest.raises(ValueError, match="task_ids must be unique"):
        ExternalEvalRunInfo(
            model_id=123,
            task_ids=["inspect_evals/aime25", "inspect_evals/aime25"],
        )


@pytest.mark.parametrize(
    ("model_or_run", "run_info", "message"),
    [
        (
            ModelOrRun.MODEL,
            ExternalEvalRunInfo(
                source_run_id="unlearning-run-id", task_ids=["inspect_evals/aime25"]
            ),
            "model launches require model_id",
        ),
        (
            ModelOrRun.RUN,
            ExternalEvalRunInfo(model_id=123, task_ids=["inspect_evals/aime25"]),
            "run launches require source_run_id",
        ),
    ],
)
def test_launch_external_eval_run_requires_source_for_endpoint(
    model_or_run: ModelOrRun,
    run_info: ExternalEvalRunInfo,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ExternalEval.launch_eval_run(model_or_run, run_info)


def test_external_eval_run_info_rejects_noncanonical_task_ids() -> None:
    with pytest.raises(ValueError, match="String should match pattern"):
        ExternalEvalRunInfo(model_id=123, task_ids=["aime25"])


def test_launch_external_eval_run_requires_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hirundo.external_eval.get_headers", lambda: {})
    monkeypatch.setattr(
        "hirundo.external_eval.requests.post",
        lambda *args, **kwargs: _Response({"message": "Run launched"}),
    )

    with pytest.raises(HirundoExternalEvalError, match="run ID"):
        ExternalEval.launch_eval_run(
            ModelOrRun.MODEL,
            ExternalEvalRunInfo(
                model_id=123,
                task_ids=["inspect_evals/aime25"],
            ),
        )


def test_launch_external_eval_run_rejects_empty_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hirundo.external_eval.get_headers", lambda: {})
    monkeypatch.setattr(
        "hirundo.external_eval.requests.post",
        lambda *args, **kwargs: _Response({"message": "Run launched", "run_id": ""}),
    )

    with pytest.raises(HirundoExternalEvalError, match="run ID"):
        ExternalEval.launch_eval_run(
            ModelOrRun.MODEL,
            ExternalEvalRunInfo(model_id=123, task_ids=["inspect_evals/aime25"]),
        )


def test_canonical_task_reference_is_public_type() -> None:
    assert CanonicalTaskReference is not None


def test_check_external_eval_run_downloads_unparsed_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "inspect-run-id.zip"
    monkeypatch.setattr(
        "hirundo.external_eval.LlmBehaviorEval._check_run_by_id",
        lambda *args, **kwargs: iter(
            [
                SseRunEventData(
                    id="inspect-run-id",
                    state=RunStatus.SUCCESS,
                    result="https://example.com/inspect.zip",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "hirundo.external_eval.download_external_eval_zip",
        lambda run_id, zip_url: ExternalEvalResults(
            cached_zip_path=result_path,
            summary_brief=None,
        ),
    )

    result = ExternalEval.check_run_by_id("inspect-run-id")

    assert result.cached_zip_path == result_path


@pytest.mark.parametrize("run_id", ["", "../inspect-run-id", "/unsafe/inspect-run-id"])
def test_check_external_eval_run_rejects_unsafe_run_ids(run_id: str) -> None:
    with pytest.raises(HirundoExternalEvalError, match="filename segments"):
        ExternalEval.check_run_by_id(run_id)


def test_check_external_eval_run_stops_for_manual_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hirundo.external_eval.LlmBehaviorEval._check_run_by_id",
        lambda *args, **kwargs: iter(
            [
                SseRunEventData(
                    id="inspect-run-id",
                    state=RunStatus.AWAITING_MANUAL_APPROVAL,
                    result=None,
                )
            ]
        ),
    )

    assert (
        ExternalEval.check_run_by_id(
            "inspect-run-id",
            stop_on_manual_approval=True,
        )
        is None
    )


@pytest.mark.asyncio
async def test_acheck_external_eval_run_yields_status_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acheck_run_by_id(run_id: str, *, max_retries: int):
        assert run_id == "inspect-run-id"
        assert max_retries > 0
        yield SseRunEventData(
            id=run_id,
            state=RunStatus.PENDING,
            result=None,
        )
        yield SseRunEventData(
            id=run_id,
            state=RunStatus.SUCCESS,
            result="https://example.com/inspect.zip",
        )

    monkeypatch.setattr(
        "hirundo.external_eval.LlmBehaviorEval.acheck_run_by_id",
        fake_acheck_run_by_id,
    )

    events = [event async for event in ExternalEval.acheck_run_by_id("inspect-run-id")]

    assert [event.state for event in events] == [
        RunStatus.PENDING,
        RunStatus.SUCCESS,
    ]


@pytest.mark.asyncio
async def test_acheck_external_eval_run_propagates_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_acheck_run_by_id(*args: object, **kwargs: object):
        assert kwargs["max_retries"] == 3
        raise HirundoLlmBehaviorEvalError("Max retries reached")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "hirundo.external_eval.LlmBehaviorEval.acheck_run_by_id",
        failing_acheck_run_by_id,
    )

    with pytest.raises(HirundoExternalEvalError, match="Max retries reached"):
        async for _ in ExternalEval.acheck_run_by_id("inspect-run-id", max_retries=3):
            pass
