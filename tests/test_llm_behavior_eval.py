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
    ReasoningEffort,
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
    # An unset reasoning level is omitted so older servers accept the request.
    assert "reasoning_effort" not in captured_request["json"]


@pytest.mark.parametrize(
    "reasoning_effort", [ReasoningEffort.DEFAULT, ReasoningEffort.NONE]
)
def test_launch_eval_run_serializes_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: ReasoningEffort,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        captured_request["json"] = kwargs["json"]
        return _Response()

    monkeypatch.setattr("hirundo.llm_behavior_eval.get_headers", lambda: {})
    monkeypatch.setattr("hirundo.llm_behavior_eval.requests.post", fake_post)

    LlmBehaviorEval.launch_eval_run(
        ModelOrRun.MODEL,
        EvalRunInfo(
            model_id=123,
            preset_type=PresetType.HALU_EVAL,
            reasoning_effort=reasoning_effort,
        ),
    )

    assert captured_request["json"]["reasoning_effort"] == reasoning_effort.value


@pytest.mark.parametrize(
    "reasoning_effort",
    [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH],
)
def test_eval_run_info_rejects_effort_levels_for_behavior_evaluations(
    reasoning_effort: ReasoningEffort,
) -> None:
    with pytest.raises(ValueError, match="only support the `default` and `none`"):
        EvalRunInfo(
            model_id=123,
            preset_type=PresetType.HALU_EVAL,
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.parametrize(
    "generation_settings",
    [
        {"attempt_timeout": 3600, "max_retries": 0, "max_model_len": 32768},
        {"attempt_timeout": None, "max_retries": None, "max_model_len": None},
        {},
    ],
)
def test_parse_inspect_evaluation_run(
    generation_settings: dict[str, int | None],
) -> None:
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
            **generation_settings,
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
    assert run_record.attempt_timeout == generation_settings.get("attempt_timeout")
    assert run_record.max_retries == generation_settings.get("max_retries")
    assert run_record.max_model_len == generation_settings.get("max_model_len")
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
    assert run_record.attempt_timeout is None
    assert run_record.max_retries is None
    assert run_record.max_model_len is None


def test_parse_legacy_evaluation_run_defaults_null_framework() -> None:
    run_record = LlmBehaviorEval._parse_eval_run_record(
        {
            "id": 1,
            "name": "Legacy evaluation",
            "model_id": None,
            "model": None,
            "source_run_id": None,
            "source_run": None,
            "framework": None,
            "run_id": "eval-run-id",
            "mlflow_run_id": None,
            "status": "SUCCESS",
            "created_at": "2026-09-08T00:00:00Z",
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
    monkeypatch.setattr("hirundo.llm_behavior_eval.time.sleep", lambda _: None)

    events = list(LlmBehaviorEval._check_run_by_id("eval-run-id"))

    assert [event.state for event in events] == [RunStatus.RETRY, RunStatus.SUCCESS]


def test_check_run_reconnects_after_manual_approval_when_not_stopping(
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
            [SseEvent(RunStatus.AWAITING_MANUAL_APPROVAL)],
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
    monkeypatch.setattr("hirundo.llm_behavior_eval.time.sleep", lambda _: None)

    events = list(LlmBehaviorEval._check_run_by_id("eval-run-id"))

    assert [event.state for event in events] == [
        RunStatus.AWAITING_MANUAL_APPROVAL,
        RunStatus.SUCCESS,
    ]


@pytest.mark.parametrize(
    ("reasoning_payload", "expected_reasoning_effort"),
    [
        ({"reasoning_effort": "none"}, ReasoningEffort.NONE),
        ({"reasoning_effort": "high"}, ReasoningEffort.HIGH),
        ({"reasoning_effort": None}, ReasoningEffort.DEFAULT),
        # Servers that predate the setting omit it.
        ({}, ReasoningEffort.DEFAULT),
    ],
)
def test_parse_eval_run_reasoning_effort(
    reasoning_payload: dict[str, str | None],
    expected_reasoning_effort: ReasoningEffort,
) -> None:
    run_record = LlmBehaviorEval._parse_eval_run_record(
        {
            "id": 1,
            "name": "evaluation",
            "run_id": "eval-run-id",
            "framework": "inspect-evals",
            "task_ids": ["inspect_evals/aime25"],
            "mlflow_run_id": None,
            "status": "SUCCESS",
            "created_at": "2026-09-29T00:00:00Z",
            **reasoning_payload,
        }
    )

    assert run_record.reasoning_effort is expected_reasoning_effort
