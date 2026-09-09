from typing import TypedDict

import pytest
from hirundo import (
    BBQBiasType,
    EvalRunInfo,
    JudgeModel,
    LlmBehaviorEval,
    ModelOrRun,
    PresetType,
)
from pydantic import JsonValue


class _Response:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {"run_id": "eval-run-id"}

    def raise_for_status(self) -> None:
        return None


class CapturedRequest(TypedDict, total=False):
    url: str
    json: dict[str, JsonValue]


def test_launch_eval_run_omits_removed_custom_dataset_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: CapturedRequest = {}

    def fake_post(
        url: str, *, json: dict[str, JsonValue], headers: dict[str, str], timeout: float
    ) -> _Response:
        captured_request["url"] = url
        captured_request["json"] = json
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
    assert "url" in captured_request
    assert "json" in captured_request
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
