import json
from typing import TypeAlias, TypedDict

import pytest
from hirundo._http import requests
from hirundo.unlearning_llm import (
    BiasBehavior,
    CustomUtility,
    HuggingFaceDataset,
    LlmRunInfo,
    LlmUnlearningRun,
    RefusalBehavior,
)
from pydantic import JsonValue, ValidationError
from requests import Response

JsonObject: TypeAlias = dict[str, JsonValue]


class ConfigRequestOptions(TypedDict):
    timeout: float


def _response(status_code: int, payload: JsonValue) -> Response:
    response = Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode()
    response.headers["Content-Type"] = "application/json"
    return response


def test_refusal_behavior_serializes_to_platform_contract() -> None:
    behavior = RefusalBehavior()

    assert behavior.model_dump(mode="json") == {"type": "REFUSAL"}
    assert LlmRunInfo.model_validate(
        {"target_behaviors": [{"type": "REFUSAL"}]}
    ).target_behaviors == [behavior]


@pytest.mark.parametrize("unsupported_field", ["aggressiveness", "biased_dataset"])
def test_refusal_behavior_rejects_unsupported_fields(
    unsupported_field: str,
) -> None:
    with pytest.raises(ValidationError):
        RefusalBehavior.model_validate(
            {"type": "REFUSAL", unsupported_field: "unsupported"}
        )


def test_refusal_run_rejects_nonempty_target_utilities() -> None:
    utility = CustomUtility(
        dataset=HuggingFaceDataset(hugging_face_dataset_name="org/dataset")
    )

    with pytest.raises(ValidationError, match="does not support target utilities"):
        LlmRunInfo(
            target_behaviors=[RefusalBehavior()],
            target_utilities=[utility],
        )


def test_refusal_launch_payload_includes_empty_target_utilities() -> None:
    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[RefusalBehavior()])
    )

    assert payload["target_behaviors"] == [{"type": "REFUSAL"}]
    assert payload["target_utilities"] == []


@pytest.mark.parametrize(
    ("config_payload", "expected_enabled"),
    [
        ({"refusalUnlearningEnabled": True}, True),
        ({"refusalUnlearningEnabled": False}, False),
        ({}, False),
    ],
)
def test_refusal_capability_uses_deployment_config(
    monkeypatch: pytest.MonkeyPatch,
    config_payload: JsonObject,
    expected_enabled: bool,
) -> None:
    request_arguments: list[tuple[str, ConfigRequestOptions]] = []

    def get_config(url: str, *, timeout: float) -> Response:
        request_arguments.append((url, {"timeout": timeout}))
        return _response(200, config_payload)

    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.get",
        get_config,
    )

    capabilities = LlmUnlearningRun.get_capabilities()

    assert capabilities.refusal_unlearning_enabled is expected_enabled
    assert "headers" not in request_arguments[0][1]


def test_older_server_config_error_uses_typed_http_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_missing_config(url: str, *, timeout: float) -> Response:
        return _response(404, {"detail": "Not Found"})

    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.get",
        get_missing_config,
    )

    with pytest.raises(requests.HTTPError, match="Not Found"):
        LlmUnlearningRun.get_capabilities()


def test_disabled_refusal_launch_uses_typed_http_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post_disabled_refusal_launch(
        url: str,
        *,
        json: JsonObject,
        headers: dict[str, str],
        timeout: float,
    ) -> Response:
        return _response(400, {"detail": "Refusal unlearning is disabled"})

    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.post",
        post_disabled_refusal_launch,
    )

    with pytest.raises(requests.HTTPError, match="Refusal unlearning is disabled"):
        LlmUnlearningRun.launch(
            model_id=1,
            run_info=LlmRunInfo(target_behaviors=[RefusalBehavior()]),
        )


def test_existing_behavior_launch_payload_is_unchanged() -> None:
    utility = CustomUtility(
        dataset=HuggingFaceDataset(hugging_face_dataset_name="org/dataset")
    )

    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[BiasBehavior()], target_utilities=[utility])
    )

    assert payload["target_behaviors"] == [{"type": "BIAS", "bias_type": "ALL"}]
    assert payload["target_utilities"] == [
        {
            "dataset": {
                "type": "HuggingFaceDataset",
                "hugging_face_dataset_name": "org/dataset",
            }
        }
    ]
