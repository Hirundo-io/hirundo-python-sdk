import json
from typing import Any

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
from pydantic import ValidationError
from requests import Response


def _response(status_code: int, payload: Any) -> Response:
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


def test_refusal_launch_payload_omits_target_utilities() -> None:
    payload = LlmUnlearningRun._build_launch_payload(
        LlmRunInfo(target_behaviors=[RefusalBehavior()])
    )

    assert payload["target_behaviors"] == [{"type": "REFUSAL"}]
    assert "target_utilities" not in payload


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
    config_payload: dict[str, object],
    expected_enabled: bool,
) -> None:
    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.get",
        lambda *args, **kwargs: _response(200, config_payload),
    )

    capabilities = LlmUnlearningRun.get_capabilities()

    assert capabilities.refusal_unlearning_enabled is expected_enabled


def test_older_server_config_error_uses_typed_http_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.get",
        lambda *args, **kwargs: _response(404, {"detail": "Not Found"}),
    )

    with pytest.raises(requests.HTTPError, match="Not Found"):
        LlmUnlearningRun.get_capabilities()


def test_disabled_refusal_launch_uses_typed_http_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hirundo.unlearning_llm.requests.post",
        lambda *args, **kwargs: _response(
            400, {"detail": "Refusal unlearning is disabled"}
        ),
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
