from typing import Any

import pytest
from hirundo import (
    ExternalEval,
    ExternalEvalRunInfo,
    HirundoExternalEvalError,
    ModelOrRun,
)


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

    run_id = ExternalEval.launch_eval_run(
        ModelOrRun.RUN,
        ExternalEvalRunInfo(
            name="Inspect AIME",
            source_run_id="unlearning-run-id",
            task_ids=["inspect_evals/aime25"],
        ),
    )

    assert run_id == "inspect-run-id"
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


def test_external_eval_run_info_rejects_duplicate_tasks() -> None:
    with pytest.raises(ValueError, match="task_ids must be unique"):
        ExternalEvalRunInfo(
            model_id=123,
            task_ids=["inspect_evals/aime25", "inspect_evals/aime25"],
        )


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
