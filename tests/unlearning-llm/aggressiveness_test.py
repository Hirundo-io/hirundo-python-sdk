import pytest
from hirundo import Aggressiveness
from hirundo.unlearning_llm import (
    BiasBehavior,
    CustomBehavior,
    HallucinationBehavior,
    HallucinationType,
    HirundoCSVDataset,
    LlmRunInfo,
    LlmUnlearningRun,
    OutputUnlearningLlmRun,
    RefusalBehavior,
    SecurityBehavior,
)
from pydantic import TypeAdapter, ValidationError


def _output_run_payload(aggressiveness: float) -> dict[str, object]:
    return {
        "id": 1,
        "name": "aggressive run",
        "model_id": 2,
        "model": {},
        "target_behaviors": [{"type": "SECURITY"}],
        "target_utilities": [],
        "advanced_options": None,
        "aggressiveness": aggressiveness,
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


@pytest.mark.parametrize("aggressiveness", [0.001, 0.5, 1.0])
def test_aggressiveness_accepts_supported_values(aggressiveness: float) -> None:
    assert TypeAdapter(Aggressiveness).validate_python(aggressiveness) == aggressiveness


@pytest.mark.parametrize("aggressiveness", [0, -0.001, 1.001, 0.0001, 0.1234])
def test_aggressiveness_rejects_unsupported_values(aggressiveness: float) -> None:
    with pytest.raises(ValidationError):
        LlmRunInfo(target_behaviors=[BiasBehavior()], aggressiveness=aggressiveness)


@pytest.mark.parametrize(
    "target_behavior",
    [
        BiasBehavior(),
        HallucinationBehavior(hallucination_type=HallucinationType.GENERAL),
        SecurityBehavior(),
        CustomBehavior(
            biased_dataset=HirundoCSVDataset(csv_url="https://example.com/bias.csv"),
            unbiased_dataset=HirundoCSVDataset(
                csv_url="https://example.com/unbiased.csv"
            ),
        ),
    ],
)
def test_non_refusal_launch_payload_includes_aggressiveness(
    target_behavior: object,
) -> None:
    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[target_behavior], aggressiveness=0.5)  # type: ignore[list-item]
    )

    assert payload["aggressiveness"] == 0.5


def test_launch_payload_omits_aggressiveness_when_unspecified() -> None:
    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[SecurityBehavior()])
    )

    assert "aggressiveness" not in payload


def test_launch_payload_revalidates_mutated_aggressiveness() -> None:
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()], aggressiveness=0.5)
    run_info.aggressiveness = 1.001

    with pytest.raises(ValidationError):
        LlmUnlearningRun._build_launch_payload(run_info)


def test_launch_payload_revalidates_mutated_behaviors() -> None:
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()], aggressiveness=0.5)
    run_info.target_behaviors.append(RefusalBehavior())

    with pytest.raises(ValidationError, match="does not support aggressiveness"):
        LlmUnlearningRun._build_launch_payload(run_info)


@pytest.mark.parametrize(
    "target_behaviors",
    [
        [RefusalBehavior()],
        [BiasBehavior(), RefusalBehavior()],
    ],
)
def test_refusal_requests_reject_aggressiveness(
    target_behaviors: list[object],
) -> None:
    with pytest.raises(ValidationError, match="does not support aggressiveness"):
        LlmRunInfo(
            target_behaviors=target_behaviors,  # type: ignore[arg-type]
            aggressiveness=0.5,
        )


def test_output_run_deserializes_aggressiveness() -> None:
    run = OutputUnlearningLlmRun.model_validate(_output_run_payload(0.75))

    assert run.aggressiveness == 0.75


def test_output_run_tolerates_legacy_aggressiveness() -> None:
    run = OutputUnlearningLlmRun.model_validate(_output_run_payload(1.001))

    assert run.aggressiveness == 1.001
