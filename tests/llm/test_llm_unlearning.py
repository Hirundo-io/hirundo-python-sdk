from __future__ import annotations

import uuid
from typing import Any

import pytest
from hirundo.llm import (
    BiasRunInfo,
    BiasType,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmUnlearningData,
    LlmUnlearningRun,
    UnlearningExample,
)

TEST_TOKEN = uuid.uuid4().hex


class DummyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.reason = None

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError("HTTP error raised during test")


@pytest.fixture(autouse=True)
def patch_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hirundo.llm.get_headers", lambda: {"Authorization": "Bearer test"})


def test_llm_model_create_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("hirundo.llm.API_HOST", "https://example.com")

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, Any], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"id": 123})

    monkeypatch.setattr("hirundo.llm.requests.post", fake_post)

    model = LlmModel(
        model_name="Example",
        model_source=HuggingFaceTransformersModel(
            model_name="huggingface-org/huggingface-id",
            token=TEST_TOKEN,
        ),
    )

    model_id = model.create()

    assert model_id == 123
    assert model.id == 123
    assert captured["url"] == "https://example.com/llm/model/"
    assert captured["json"]["model_name"] == "Example"
    assert captured["json"]["model_source"]["model_source_type"] == "HUGGING_FACE_TRANSFORMERS"
    assert captured["headers"] == {"Authorization": "Bearer test"}
    assert captured["timeout"] > 0


def test_unlearning_data_create_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("hirundo.llm.API_HOST", "https://example.com")

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, Any], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"id": 456})

    monkeypatch.setattr("hirundo.llm.requests.post", fake_post)

    unlearning_data = LlmUnlearningData(
        name="Test Data",
        description="A dataset for testing",
        examples=[
            UnlearningExample(prompt="Remove bias", completion="Acknowledged"),
        ],
        tags=["test"],
    )

    data_id = unlearning_data.create()

    assert data_id == 456
    assert unlearning_data.id == 456
    assert captured["url"] == "https://example.com/llm/unlearning-data/"
    assert captured["json"]["name"] == "Test Data"
    assert captured["json"]["examples"][0]["prompt"] == "Remove bias"
    assert captured["headers"] == {"Authorization": "Bearer test"}


def test_unlearning_run_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("hirundo.llm.API_HOST", "https://example.com")

    def fake_post(url: str, json: dict[str, Any] | None, headers: dict[str, Any], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"run_id": "run-789", "status": "PENDING"})

    monkeypatch.setattr("hirundo.llm.requests.post", fake_post)

    run_info = BiasRunInfo(bias_type=BiasType.ALL, unlearning_data_ids=[456])

    run = LlmUnlearningRun.launch(321, run_info)

    assert run.run_id == "run-789"
    assert run.status == "PENDING"
    assert captured["url"] == "https://example.com/llm/unlearning/run/321"
    assert captured["json"] == {"bias_type": "ALL", "unlearning_data_ids": [456]}
    assert captured["headers"] == {"Authorization": "Bearer test"}
