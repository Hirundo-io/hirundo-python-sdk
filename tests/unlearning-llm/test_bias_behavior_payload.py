import pytest
from hirundo import BBQBiasType
from hirundo.unlearning_llm import (
    BiasBehavior,
    CustomUtility,
    HuggingFaceDataset,
    LlmRunInfo,
    LlmUnlearningRun,
    OutputBiasBehavior,
    OutputUnlearningLlmRun,
)
from pydantic import ValidationError


def test_bias_behavior_has_no_public_bias_type() -> None:
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()])
    target_behavior = run_info.target_behaviors[0]
    assert isinstance(target_behavior, BiasBehavior)
    assert "bias_type" not in target_behavior.model_dump(mode="json")


def test_llm_run_info_forwards_target_utilities() -> None:
    utility = CustomUtility(
        dataset=HuggingFaceDataset(hugging_face_dataset_name="org/dataset")
    )
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()], target_utilities=[utility])
    assert run_info.target_utilities == [utility]


def test_bias_behavior_rejects_bias_type() -> None:
    with pytest.raises(ValidationError):
        BiasBehavior.model_validate({"type": "BIAS", "bias_type": BBQBiasType.ALL})


def test_llm_run_info_rejects_legacy_bias_dict_without_type() -> None:
    with pytest.raises(ValidationError):
        LlmRunInfo.model_validate(
            {"target_behaviors": [{"bias_type": BBQBiasType.RACE}]}
        )


def test_launch_payload_adds_backend_only_all_bias_type() -> None:
    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[BiasBehavior()])
    )

    assert payload["target_behaviors"] == [{"type": "BIAS", "bias_type": "ALL"}]


def test_output_unlearning_run_accepts_legacy_bias_subtypes() -> None:
    run = OutputUnlearningLlmRun.model_validate(
        {
            "id": 1,
            "name": "legacy run",
            "model_id": 2,
            "model": {},
            "target_behaviors": [{"type": "BIAS", "bias_type": "RACE"}],
            "target_utilities": [],
            "advanced_options": None,
            "run_id": "run-id",
            "mlflow_run_id": None,
            "status": "SUCCESS",
            "approved": True,
            "created_at": "2026-01-01T00:00:00Z",
            "completed_at": None,
            "pre_process_progress": 100.0,
            "optimization_progress": 100.0,
            "post_process_progress": 100.0,
        }
    )

    target_behavior = run.target_behaviors[0]
    assert isinstance(target_behavior, OutputBiasBehavior)
    assert target_behavior.bias_type == BBQBiasType.RACE
