import datetime
from unittest.mock import patch

from hirundo._llm_sources import (
    HuggingFaceTransformersModelOutput,
    LocalTransformersModel,
)
from hirundo.cli import app
from hirundo.unlearning_llm import LlmModelOut
from rich.console import Console
from typer.testing import CliRunner

_CREATED_AT = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)


def _llm_model(
    model_id: int,
    model_name: str,
    model_source: HuggingFaceTransformersModelOutput | LocalTransformersModel,
) -> LlmModelOut:
    return LlmModelOut(
        id=model_id,
        organization_id=7,
        creator_id=3,
        creator_name="Test User",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
        model_name=model_name,
        model_source=model_source,
    )


def test_llm_list_prints_each_model() -> None:
    llm_models = [
        _llm_model(
            11,
            "hub-llm",
            HuggingFaceTransformersModelOutput(model_name="org/hub-model"),
        ),
        _llm_model(12, "disk-llm", LocalTransformersModel(local_path="/models/x")),
    ]

    with (
        patch("hirundo.unlearning_llm.LlmModel.list", return_value=llm_models) as (
            list_mock
        ),
        patch("hirundo._cli_common.console", Console(width=200)),
    ):
        result = CliRunner().invoke(app, ["llm", "list"])

    assert result.exit_code == 0, result.output
    list_mock.assert_called_once_with(organization_id=None)
    for expected_text in (
        "11",
        "hub-llm",
        "huggingface_transformers",
        "org/hub-model",
        "12",
        "disk-llm",
        "local_transformers",
        "/models/x",
        _CREATED_AT.isoformat(),
    ):
        assert expected_text in result.output


def test_llm_list_filters_by_organization() -> None:
    with patch("hirundo.unlearning_llm.LlmModel.list", return_value=[]) as list_mock:
        result = CliRunner().invoke(app, ["llm", "list", "--organization-id", "7"])

    assert result.exit_code == 0, result.output
    list_mock.assert_called_once_with(organization_id=7)
