import secrets

import hirundo.unlearning_llm as unlearning_llm_module
import pytest
from hirundo._llm_sources import HuggingFaceTransformersModel
from hirundo.unlearning_llm import LlmModel


class _FakeResponse:
    def __init__(self, model_id: int) -> None:
        self._model_id = model_id

    def json(self) -> dict[str, int]:
        return {"id": self._model_id}


def _build_huggingface_llm_model() -> LlmModel:
    user_access_secret = secrets.token_hex(8)
    return LlmModel(
        model_name="test-llm",
        model_source=HuggingFaceTransformersModel(
            model_name="org/private-model",
            token=user_access_secret,
        ),
    )


def _stub_create_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        unlearning_llm_module.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(model_id=123),
    )
    monkeypatch.setattr(
        unlearning_llm_module,
        "raise_for_status_with_reason",
        lambda _response: None,
    )


def test_llm_model_create_skips_hf_access_validation_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls: list[tuple[str, str | None, str]] = []

    monkeypatch.setenv("HIRUNDO_VALIDATE_HF_ACCESS", "false")
    _stub_create_dependencies(monkeypatch)
    monkeypatch.setattr(
        unlearning_llm_module,
        "validate_huggingface_model_access",
        lambda model_name, token, model_role: validation_calls.append(
            (model_name, token, model_role)
        ),
    )

    _build_huggingface_llm_model().create()

    assert validation_calls == []


def test_llm_model_create_validates_hf_access_when_parameter_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls: list[tuple[str, str | None, str]] = []

    monkeypatch.setenv("HIRUNDO_VALIDATE_HF_ACCESS", "false")
    _stub_create_dependencies(monkeypatch)
    monkeypatch.setattr(
        unlearning_llm_module,
        "validate_huggingface_model_access",
        lambda model_name, token, model_role: validation_calls.append(
            (model_name, token, model_role)
        ),
    )

    llm_model = _build_huggingface_llm_model()
    llm_model.create(validate_hf_access=True)

    assert isinstance(llm_model.model_source, HuggingFaceTransformersModel)
    assert validation_calls == [
        ("org/private-model", llm_model.model_source.token, "LLM")
    ]


def test_llm_model_create_validates_hf_access_when_env_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls: list[tuple[str, str | None, str]] = []

    monkeypatch.setenv("HIRUNDO_VALIDATE_HF_ACCESS", "true")
    _stub_create_dependencies(monkeypatch)
    monkeypatch.setattr(
        unlearning_llm_module,
        "validate_huggingface_model_access",
        lambda model_name, token, model_role: validation_calls.append(
            (model_name, token, model_role)
        ),
    )

    llm_model = _build_huggingface_llm_model()
    llm_model.create()

    assert isinstance(llm_model.model_source, HuggingFaceTransformersModel)
    assert validation_calls == [
        ("org/private-model", llm_model.model_source.token, "LLM")
    ]
