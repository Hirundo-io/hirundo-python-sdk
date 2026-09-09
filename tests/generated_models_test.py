import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hirundo._generated.wire_models import (
    CreateLlm,
    ServerUnlearningLlmModelsEvalRunInfo,
)
from hirundo._llm_sources import HuggingFaceTransformersModel
from hirundo.llm_behavior_eval import (
    EvalRunInfo,
    JudgeModel,
    LlmEvalMetricRow,
    LlmEvalMetrics,
    PresetType,
)
from hirundo.llm_bias_type import BBQBiasType
from hirundo.unlearning_llm import (
    CustomBehavior,
    DatasetType,
    HirundoCSVDataset,
    HuggingFaceDataset,
    LlmModel,
    LlmModelOut,
    LlmRunInfo,
    UnlearningLlmAdvancedOptions,
)
from pydantic import ValidationError
from scripts.generate_models import DEFAULT_SCHEMA

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_generation_input_is_the_complete_openapi_document() -> None:
    schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))

    assert schema["openapi"] == "3.1.0"
    assert schema["paths"]
    assert "CreateLlm" in schema["components"]["schemas"]
    assert (
        "server__unlearning_llm__models__eval__RunInfo"
        in schema["components"]["schemas"]
    )


def test_eval_request_uses_generated_fields_and_public_nested_models() -> None:
    run_info = EvalRunInfo.model_validate(
        {
            "preset_type": "BBQ_BIAS",
            "bias_type": "ALL",
            "judge_model": {"path_or_repo_id": "org/judge"},
        }
    )

    assert isinstance(run_info.judge_model, JudgeModel)
    assert run_info.bias_type == BBQBiasType.ALL
    assert run_info.model_dump(mode="json")["preset_type"] == "BBQ_BIAS"
    generated = ServerUnlearningLlmModelsEvalRunInfo.model_validate(
        run_info.model_dump(mode="json")
    )
    assert generated.preset_type == PresetType.BBQ_BIAS


def test_eval_request_preserves_required_and_forbidden_constraints() -> None:
    with pytest.raises(ValidationError):
        EvalRunInfo.model_validate({})
    with pytest.raises(ValidationError):
        EvalRunInfo.model_validate({"preset_type": "BBQ_BIAS", "removed": True})
    with pytest.raises(ValidationError, match="refusal presets"):
        EvalRunInfo.model_validate(
            {"preset_type": PresetType.XSTEST, "bias_type": BBQBiasType.ALL}
        )


def test_judge_model_accepts_token_id_and_rejects_two_token_sources() -> None:
    judge_model = JudgeModel.model_validate(
        {"path_or_repo_id": "org/judge", "token_id": 42}
    )
    assert judge_model.token_id == 42
    with pytest.raises(ValidationError, match="Only one"):
        JudgeModel.model_validate(
            {"path_or_repo_id": "org/judge", "token": "secret", "token_id": 42}
        )


def test_eval_metrics_raw_rows_use_public_subclass_and_allow_extensions() -> None:
    metrics = LlmEvalMetrics.model_validate(
        {"rows": [{"benchmark": "bbq", "metric": "accuracy", "detail": 1}]}
    )
    assert isinstance(metrics.rows[0], LlmEvalMetricRow)
    assert metrics.model_dump()["rows"][0]["detail"] == 1


def test_llm_model_raw_source_uses_public_model_and_rejects_null_bool() -> None:
    model = LlmModel.model_validate(
        {
            "model_name": "example",
            "model_source": {
                "type": "huggingface_transformers",
                "model_name": "org/model",
            },
        }
    )
    assert isinstance(model.model_source, HuggingFaceTransformersModel)
    assert model.archive_existing_runs is True
    generated = CreateLlm.model_validate(
        {**model.model_dump(mode="json"), "replace_if_exists": False}
    )
    assert generated.model_name == "example"
    with pytest.raises(ValidationError):
        LlmModel.model_validate(
            {
                "model_name": "example",
                "model_source": {
                    "type": "huggingface_transformers",
                    "model_name": "org/model",
                },
                "archive_existing_runs": None,
            }
        )


def test_llm_output_retains_naive_datetime_compatibility() -> None:
    output = LlmModelOut.model_validate(
        {
            "id": 1,
            "organization_id": 2,
            "creator_id": 3,
            "creator_name": "creator",
            "created_at": datetime.datetime(2026, 1, 1),
            "updated_at": datetime.datetime(2026, 1, 2),
            "model_name": "example",
            "model_source": {
                "type": "huggingface_transformers",
                "model_name": "org/model",
            },
        }
    )
    assert output.created_at.tzinfo is None


def test_unlearning_raw_nested_models_and_defaults_round_trip() -> None:
    run_info = LlmRunInfo.model_validate(
        {
            "target_behaviors": [
                {
                    "type": "CUSTOM",
                    "biased_dataset": {"type": "HirundoCSV", "csv_url": "file.csv"},
                    "unbiased_dataset": {
                        "type": "HuggingFaceDataset",
                        "hugging_face_dataset_name": "org/data",
                    },
                }
            ],
            "advanced_options": {"max_tokens_for_model": {"bias": 100}},
        }
    )
    behavior = run_info.target_behaviors[0]
    assert isinstance(behavior, CustomBehavior)
    assert isinstance(behavior.biased_dataset, HirundoCSVDataset)
    assert isinstance(behavior.unbiased_dataset, HuggingFaceDataset)
    assert run_info.target_utilities == []
    assert isinstance(run_info.advanced_options, UnlearningLlmAdvancedOptions)
    assert run_info.advanced_options.max_tokens_for_model == {DatasetType.BIAS: 100}
    assert LlmRunInfo.model_validate(run_info.model_dump(mode="json")) == run_info


def test_unlearning_discriminator_rejects_unknown_behavior() -> None:
    with pytest.raises(ValidationError):
        LlmRunInfo.model_validate({"target_behaviors": [{"type": "UNKNOWN"}]})


def test_generated_output_is_reproducible() -> None:
    output_path = REPOSITORY_ROOT / "hirundo" / "_generated" / "wire_models.py"
    before = output_path.read_bytes()
    subprocess.run(  # noqa: S603 - executable is the current interpreter
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / "generate_models.py")],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    assert output_path.read_bytes() == before


def test_generated_package_is_discoverable_for_wheel_builds() -> None:
    from setuptools import find_packages

    assert "hirundo._generated" in find_packages(where=REPOSITORY_ROOT)
