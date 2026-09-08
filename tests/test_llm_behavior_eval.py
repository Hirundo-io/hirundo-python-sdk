from typing import Any

import pytest
from hirundo import (
    BBQBiasType,
    EvalFramework,
    EvalRunInfo,
    JudgeModel,
    LlmBehaviorEval,
    ModelOrRun,
    PresetType,
)
from hirundo._run_status import RunStatus
from hirundo.llm_behavior_eval import LlmEvalMetricRow, LlmEvalMetrics


class _Response:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {"run_id": "eval-run-id"}

    def raise_for_status(self) -> None:
        return None


def test_launch_eval_run_omits_removed_custom_dataset_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        captured_request["url"] = args[0]
        captured_request["json"] = kwargs["json"]
        return _Response()

    monkeypatch.setattr("hirundo.llm_behavior_eval.get_headers", lambda: {})
    monkeypatch.setattr("hirundo.llm_behavior_eval.requests.post", fake_post)

    run_id = LlmBehaviorEval.launch_eval_run(
        ModelOrRun.MODEL,
        EvalRunInfo(
            name="preset-evaluation",
            model_id=123,
            preset_type=PresetType.BBQ_BIAS,
            bias_type=BBQBiasType.ALL,
            judge_model=JudgeModel(path_or_repo_id="Qwen/Qwen3-0.6B"),
        ),
    )

    assert run_id == "eval-run-id"
    assert captured_request["url"].endswith("/llm-behavior-eval/run/model")
    assert captured_request["json"] == {
        "organization_id": None,
        "name": "preset-evaluation",
        "model_id": 123,
        "source_run_id": None,
        "preset_type": "BBQ_BIAS",
        "bias_type": "ALL",
        "judge_model": {
            "path_or_repo_id": "Qwen/Qwen3-0.6B",
            "token": None,
            "batch_size": None,
            "output_tokens": None,
            "use_4bit": None,
        },
    }
    assert "file_path" not in captured_request["json"]


def test_parse_inspect_evaluation_run() -> None:
    run_record = LlmBehaviorEval._parse_eval_run_record(
        {
            "id": 1,
            "name": "inspect-evaluation",
            "model_id": 123,
            "model": None,
            "source_run_id": None,
            "source_run": None,
            "framework": "inspect-evals",
            "preset_type": None,
            "bias_type": None,
            "task_ids": ["inspect_evals/aime25"],
            "sample_limit": None,
            "judge_model": None,
            "run_id": "eval-run-id",
            "mlflow_run_id": None,
            "status": "SUCCESS",
            "created_at": "2026-09-08T00:00:00Z",
            "pre_process_progress": 100.0,
            "optimization_progress": 100.0,
            "post_process_progress": 100.0,
            "metrics": {
                "rows": [
                    {
                        "benchmark": "AIME 2025",
                        "metric": "accuracy",
                        "score": 0.5,
                        "runtime_seconds": 12.5,
                    }
                ]
            },
        }
    )

    assert run_record.framework is EvalFramework.INSPECT_EVALS
    assert run_record.task_ids == ["inspect_evals/aime25"]
    assert run_record.sample_limit is None
    assert run_record.metrics == LlmEvalMetrics(
        rows=[
            LlmEvalMetricRow(
                benchmark="AIME 2025",
                metric="accuracy",
                score=0.5,
                runtime_seconds=12.5,
            )
        ]
    )


def test_parse_legacy_evaluation_run_defaults_framework() -> None:
    run_record = LlmBehaviorEval._parse_eval_run_record(
        {
            "id": 1,
            "name": "legacy-evaluation",
            "model_id": 123,
            "model": None,
            "source_run_id": None,
            "source_run": None,
            "preset_type": "BBQ_BIAS",
            "bias_type": "ALL",
            "judge_model": None,
            "run_id": "eval-run-id",
            "mlflow_run_id": None,
            "status": "SUCCESS",
            "created_at": "2026-09-08T00:00:00Z",
            "pre_process_progress": 100.0,
            "optimization_progress": 100.0,
            "post_process_progress": 100.0,
        }
    )

    assert run_record.framework is EvalFramework.LLM_BEHAVIOR_EVAL


def test_check_run_reconnects_after_retry_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SseEvent:
        event = ""

        def __init__(self, state: RunStatus):
            self.data = (
                '{"data":{"id":"eval-run-id","state":"'
                + state.value
                + '","result":null}}'
            )

    class Client:
        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    event_streams = iter(
        [
            [SseEvent(RunStatus.RETRY)],
            [SseEvent(RunStatus.SUCCESS)],
        ]
    )
    monkeypatch.setattr(
        "hirundo.llm_behavior_eval.httpx.Client", lambda **kwargs: Client()
    )
    monkeypatch.setattr(
        "hirundo.llm_behavior_eval.iter_sse_retrying",
        lambda *args, **kwargs: iter(next(event_streams)),
    )
    monkeypatch.setattr("hirundo.llm_behavior_eval.get_headers", lambda: {})

    events = list(LlmBehaviorEval._check_run_by_id("eval-run-id"))

    assert [event.state for event in events] == [RunStatus.RETRY, RunStatus.SUCCESS]
