import io
import zipfile
from pathlib import Path
from types import TracebackType
from typing import TypedDict

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

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.payload]


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token = "".join(["<", "eos", ">"])


class FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def to_dict(self) -> dict[str, str]:
        return {"model_type": self.model_type}


class FakeModel:
    """Model identity returned by the patched model loaders."""


class FakePipeline:
    """Pipeline identity returned by the patched pipeline factory."""


class LoaderKeywords(TypedDict):
    token: str | None
    trust_remote_code: bool


class LoaderCall(TypedDict):
    args: tuple[str]
    kwargs: LoaderKeywords


class PipelineCall(TypedDict):
    task: str
    model: FakeModel
    tokenizer: FakeTokenizer
    config: FakeConfig
    device: str | int | None
    device_map: str | dict[str, int | str] | None


def _loader_call(
    model_name: str, token: str | None, trust_remote_code: bool
) -> LoaderCall:
    """Record the arguments passed to a Transformers pretrained loader.

    Args:
        model_name: Source model repository or local path.
        token: Optional repository access token.
        trust_remote_code: Whether the loader may load custom model code.

    Returns:
        The positional and keyword arguments used by the SDK.
    """
    return {
        "args": (model_name,),
        "kwargs": {"token": token, "trust_remote_code": trust_remote_code},
    }


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
    base_model = FakeModel()
    peft_model = FakeModel()
    pipeline_result = FakePipeline()
    tokenizer_calls: list[LoaderCall] = []
    config_calls: list[LoaderCall] = []
    causal_lm_calls: list[LoaderCall] = []
    peft_calls: list[tuple[FakeModel, str]] = []
    pipeline_calls: list[PipelineCall] = []

    def fake_requests_get(url: str, *, timeout: float, stream: bool) -> FakeResponse:
        return FakeResponse(adapter_zip_bytes)

    def fake_tokenizer_from_pretrained(
        model_name: str, *, token: str | None, trust_remote_code: bool
    ) -> FakeTokenizer:
        tokenizer_calls.append(_loader_call(model_name, token, trust_remote_code))
        return tokenizer

    def fake_config_from_pretrained(
        model_name: str, *, token: str | None, trust_remote_code: bool
    ) -> FakeConfig:
        config_calls.append(_loader_call(model_name, token, trust_remote_code))
        return config

    def fake_causal_lm_from_pretrained(
        model_name: str, *, token: str | None, trust_remote_code: bool
    ) -> FakeModel:
        causal_lm_calls.append(_loader_call(model_name, token, trust_remote_code))
        return base_model

    def fake_peft_from_pretrained(model: FakeModel, path: str) -> FakeModel:
        peft_calls.append((model, path))
        return peft_model

    def fake_pipeline(
        *,
        task: str,
        model: FakeModel,
        tokenizer: FakeTokenizer,
        config: FakeConfig,
        device: str | int | None,
        device_map: str | dict[str, int | str] | None,
    ) -> FakePipeline:
        pipeline_calls.append(
            {
                "task": task,
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "device": device,
                "device_map": device_map,
            }
        )
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
    assert Path(peft_calls[0][1]).name == "unlearned_model_folder"
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
    multimodal_base_model = FakeModel()
    peft_model = FakeModel()
    pipeline_result = FakePipeline()
    multimodal_calls: list[LoaderCall] = []

    def fake_requests_get(url: str, *, timeout: float, stream: bool) -> FakeResponse:
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

    def fake_multimodal_from_pretrained(
        model_name: str, *, token: str | None, trust_remote_code: bool
    ) -> FakeModel:
        multimodal_calls.append(_loader_call(model_name, token, trust_remote_code))
        return multimodal_base_model

    monkeypatch.setattr(
        AutoModelForImageTextToText, "from_pretrained", fake_multimodal_from_pretrained
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
