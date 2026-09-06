import datetime
from typing import Any

import pytest
from hirundo import BBQBiasType, EvalRunInfo, LlmBehaviorEval, ModelOrRun, PresetType
from pydantic import ValidationError


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


@pytest.mark.parametrize(
    ("preset_type", "model_or_run", "model_id", "source_run_id"),
    [
        (PresetType.XSTEST, ModelOrRun.MODEL, 12, None),
        (PresetType.OR_BENCH, ModelOrRun.RUN, None, "run-12"),
    ],
)
def test_refusal_preset_launch_serializes_platform_contract(
    monkeypatch: pytest.MonkeyPatch,
    preset_type: PresetType,
    model_or_run: ModelOrRun,
    model_id: int | None,
    source_run_id: str | None,
) -> None:
    request_payloads: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs["json"])
        return _Response({"run_id": "eval-run-id"})

    monkeypatch.setattr("hirundo.llm_behavior_eval.requests.post", fake_post)
    run_info = EvalRunInfo(
        name="Refusal evaluation",
        preset_type=preset_type,
        model_id=model_id,
        source_run_id=source_run_id,
    )

    run_id = LlmBehaviorEval.launch_eval_run(model_or_run, run_info)

    assert run_id == "eval-run-id"
    assert request_payloads[0]["preset_type"] == preset_type.value
    assert request_payloads[0]["model_id"] == model_id
    assert request_payloads[0]["source_run_id"] == source_run_id
    assert request_payloads[0]["bias_type"] is None


@pytest.mark.parametrize("preset_type", [PresetType.XSTEST, PresetType.OR_BENCH])
def test_refusal_preset_response_is_typed(preset_type: PresetType) -> None:
    record = LlmBehaviorEval._parse_eval_run_record(
        {
            "id": 1,
            "name": "Refusal evaluation",
            "model_id": 12,
            "preset_type": preset_type.value,
            "bias_type": None,
            "run_id": "eval-run-id",
            "status": "PENDING",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )

    assert record.preset_type is preset_type
    assert record.bias_type is None


@pytest.mark.parametrize("preset_type", [PresetType.XSTEST, PresetType.OR_BENCH])
def test_refusal_preset_rejects_bias_type(preset_type: PresetType) -> None:
    with pytest.raises(ValidationError, match="not supported for refusal presets"):
        EvalRunInfo(preset_type=preset_type, bias_type=BBQBiasType.ALL)


def test_existing_preset_serialization_is_unchanged() -> None:
    run_info = EvalRunInfo(
        model_id=12,
        preset_type=PresetType.BBQ_BIAS,
        bias_type=BBQBiasType.ALL,
    )

    assert run_info.model_dump(mode="json")["preset_type"] == "BBQ_BIAS"
    assert run_info.model_dump(mode="json")["bias_type"] == "ALL"


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalRunInfo.model_validate({"preset_type": "UNKNOWN"})
