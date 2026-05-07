import io
import zipfile
from typing import Any

import pytest
from hirundo._llm_pipeline import get_hf_pipeline_for_run_given_model
from hirundo._llm_sources import HuggingFaceTransformersModel
from hirundo.unlearning_llm import LlmModel, LlmUnlearningRun

pytest.importorskip("peft")
pytest.importorskip("transformers")


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exception_type, exception_value, traceback_value) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.payload]


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "".join(["<", "eos", ">"])


class FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def to_dict(self) -> dict[str, str]:
        return {"model_type": self.model_type}


@pytest.fixture
def adapter_zip_bytes() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("unlearned_model_folder/adapter_config.json", "{}")
    return zip_buffer.getvalue()


def test_text_generation_pipeline_uses_transformers_loader_api(
    monkeypatch: pytest.MonkeyPatch, adapter_zip_bytes: bytes
) -> None:
    tokenizer = FakeTokenizer()
    config = FakeConfig(model_type="not-multimodal")
    base_model = object()
    peft_model = object()
    pipeline_result = object()
    tokenizer_calls: list[dict[str, Any]] = []
    config_calls: list[dict[str, Any]] = []
    causal_lm_calls: list[dict[str, Any]] = []
    peft_calls: list[tuple[object, str]] = []
    pipeline_calls: list[dict[str, Any]] = []

    def fake_requests_get(*args: Any, **kwargs: Any) -> FakeResponse:
        del args, kwargs
        return FakeResponse(adapter_zip_bytes)

    def fake_tokenizer_from_pretrained(*args: Any, **kwargs: Any) -> FakeTokenizer:
        tokenizer_calls.append({"args": args, "kwargs": kwargs})
        return tokenizer

    def fake_config_from_pretrained(*args: Any, **kwargs: Any) -> FakeConfig:
        config_calls.append({"args": args, "kwargs": kwargs})
        return config

    def fake_causal_lm_from_pretrained(*args: Any, **kwargs: Any) -> object:
        causal_lm_calls.append({"args": args, "kwargs": kwargs})
        return base_model

    def fake_peft_from_pretrained(model: object, path: str) -> object:
        peft_calls.append((model, path))
        return peft_model

    def fake_pipeline(**kwargs: Any) -> object:
        pipeline_calls.append(kwargs)
        return pipeline_result

    monkeypatch.setattr(
        LlmUnlearningRun,
        "check_run_by_id",
        lambda run_id: {"result": "https://example.invalid/adapter.zip"},
    )
    monkeypatch.setattr("hirundo._llm_pipeline.requests.get", fake_requests_get)

    from peft import PeftModel
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_config_from_pretrained)
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", fake_causal_lm_from_pretrained
    )
    monkeypatch.setattr(PeftModel, "from_pretrained", fake_peft_from_pretrained)
    monkeypatch.setattr("transformers.pipelines.pipeline", fake_pipeline)
    monkeypatch.setattr("hirundo._llm_pipeline.pipeline", fake_pipeline, raising=False)

    llm = LlmModel(
        model_name="demo-model",
        model_source=HuggingFaceTransformersModel(
            model_name="org/demo-model",
            token="-".join(["hf", "token"]),
        ),
    )

    pipeline_output = get_hf_pipeline_for_run_given_model(
        llm,
        "run-123",
        device="cpu",
        device_map="auto",
        trust_remote_code=True,
    )

    assert pipeline_output is pipeline_result
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer_calls == [
        {
            "args": ("org/demo-model",),
            "kwargs": {
                "token": "-".join(["hf", "token"]),
                "trust_remote_code": True,
            },
        }
    ]
    assert config_calls == [
        {
            "args": ("org/demo-model",),
            "kwargs": {
                "token": "-".join(["hf", "token"]),
                "trust_remote_code": True,
            },
        }
    ]
    assert causal_lm_calls == [
        {
            "args": ("org/demo-model",),
            "kwargs": {
                "token": "-".join(["hf", "token"]),
                "trust_remote_code": True,
            },
        }
    ]
    assert peft_calls and peft_calls[0][0] is base_model
    assert peft_calls[0][1].endswith("/unlearned_model_folder")
    assert pipeline_calls == [
        {
            "task": "text-generation",
            "model": peft_model,
            "tokenizer": tokenizer,
            "config": config,
            "device": "cpu",
            "device_map": "auto",
        }
    ]


def test_multimodal_pipeline_uses_image_text_loader_when_model_type_matches(
    monkeypatch: pytest.MonkeyPatch, adapter_zip_bytes: bytes
) -> None:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    multimodal_model_type = next(iter(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES))
    tokenizer = FakeTokenizer()
    config = FakeConfig(model_type=multimodal_model_type)
    multimodal_base_model = object()
    peft_model = object()
    pipeline_result = object()
    multimodal_calls: list[dict[str, Any]] = []

    def fake_requests_get(*args: Any, **kwargs: Any) -> FakeResponse:
        del args, kwargs
        return FakeResponse(adapter_zip_bytes)

    monkeypatch.setattr(
        LlmUnlearningRun,
        "check_run_by_id",
        lambda run_id: {"result": "https://example.invalid/adapter.zip"},
    )
    monkeypatch.setattr("hirundo._llm_pipeline.requests.get", fake_requests_get)

    from peft import PeftModel
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (
            multimodal_calls.append({"args": args, "kwargs": kwargs})
            or multimodal_base_model
        ),
    )
    monkeypatch.setattr(PeftModel, "from_pretrained", lambda model, path: peft_model)
    monkeypatch.setattr(
        "transformers.pipelines.pipeline", lambda **kwargs: pipeline_result
    )
    monkeypatch.setattr(
        "hirundo._llm_pipeline.pipeline",
        lambda **kwargs: pipeline_result,
        raising=False,
    )

    llm = LlmModel(
        model_name="demo-model",
        model_source=HuggingFaceTransformersModel(model_name="org/demo-model"),
    )

    pipeline_output = get_hf_pipeline_for_run_given_model(llm, "run-123")

    assert pipeline_output is pipeline_result
    assert multimodal_calls == [
        {
            "args": ("org/demo-model",),
            "kwargs": {"token": None, "trust_remote_code": False},
        }
    ]
