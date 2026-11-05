import json
from collections.abc import Callable

import pytest
from hirundo import (
    EvalRunInfo,
    GitRepo,
    HirundoError,
    LlmBehaviorEval,
    LlmModel,
    LlmRunInfo,
    LlmUnlearningRun,
    ModelOrRun,
    PresetType,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)
from hirundo._llm_sources import HuggingFaceTransformersModel
from hirundo.llm_behavior_eval import HirundoLlmBehaviorEvalError
from hirundo.unlearning_llm import SecurityBehavior
from pydantic import JsonValue
from requests import Response
from tests.api_response_payloads import (
    create_response_payload,
    launch_response_payload,
)

CREATED_ID = 42
RUN_ID = "launched-run-id"


def _response(body: bytes) -> Response:
    response = Response()
    response.status_code = 200
    response._content = body
    response.headers["Content-Type"] = "application/json"
    return response


def _json_body(payload: JsonValue) -> bytes:
    return json.dumps(payload).encode()


def _patch_post(monkeypatch: pytest.MonkeyPatch, module: str, body: bytes) -> None:
    def fake_post(
        url: str,
        *,
        json: dict[str, JsonValue],
        headers: dict[str, str],
        timeout: float,
    ) -> Response:
        return _response(body)

    monkeypatch.setattr(f"hirundo.{module}.get_headers", lambda: {})
    monkeypatch.setattr(f"hirundo.{module}.requests.post", fake_post)


def _llm_model() -> LlmModel:
    return LlmModel(
        model_name="response-validation-llm",
        model_source=HuggingFaceTransformersModel(model_name="org/base-model"),
    )


def _git_repo() -> GitRepo:
    return GitRepo(
        name="response-validation-repo",
        repository_url="https://example.com/org/repo.git",
    )


def _storage_config() -> StorageConfig:
    return StorageConfig(
        name="response-validation-storage",
        type=StorageTypes.GCP,
        gcp=StorageGCP(bucket_name="bucket", project="project"),
    )


CreatableResource = LlmModel | GitRepo | StorageConfig

CREATE_ENDPOINTS: list[tuple[str, Callable[[], CreatableResource]]] = [
    ("unlearning_llm", _llm_model),
    ("git", _git_repo),
    ("storage", _storage_config),
]
CREATE_ENDPOINT_IDS = ["llm-model", "git-repo", "storage-config"]


@pytest.mark.parametrize(
    ("module", "build_resource"), CREATE_ENDPOINTS, ids=CREATE_ENDPOINT_IDS
)
def test_create_returns_id_from_generated_create_model(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    build_resource: Callable[[], CreatableResource],
) -> None:
    _patch_post(monkeypatch, module, _json_body(create_response_payload(CREATED_ID)))
    resource = build_resource()

    assert resource.create() == CREATED_ID
    assert resource.id == CREATED_ID


@pytest.mark.parametrize(
    ("module", "build_resource"), CREATE_ENDPOINTS, ids=CREATE_ENDPOINT_IDS
)
@pytest.mark.parametrize(
    "body",
    [
        _json_body({"message": "Created"}),
        _json_body(create_response_payload(None)),
        _json_body(create_response_payload("")),
        _json_body(CREATED_ID),
        b"",
    ],
    ids=["missing-id", "null-id", "empty-id", "bare-id", "empty-body"],
)
def test_create_rejects_response_without_valid_id(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    build_resource: Callable[[], CreatableResource],
    body: bytes,
) -> None:
    _patch_post(monkeypatch, module, body)
    resource = build_resource()

    with pytest.raises(HirundoError, match="invalid CreateModel response"):
        resource.create()
    assert resource.id is None


def _launch_unlearning_run() -> str:
    return LlmUnlearningRun.launch(
        model_id=7, run_info=LlmRunInfo(target_behaviors=[SecurityBehavior()])
    )


def _launch_behavior_eval_run() -> str:
    return LlmBehaviorEval.launch_eval_run(
        ModelOrRun.MODEL,
        EvalRunInfo(model_id=7, preset_type=PresetType.BBQ_BIAS),
    )


LAUNCH_ENDPOINTS: list[tuple[str, Callable[[], str], type[HirundoError]]] = [
    ("unlearning_llm", _launch_unlearning_run, HirundoError),
    ("llm_behavior_eval", _launch_behavior_eval_run, HirundoLlmBehaviorEvalError),
]


@pytest.mark.parametrize(
    ("module", "launch", "error_cls"),
    LAUNCH_ENDPOINTS,
    ids=["unlearning-run", "behavior-eval-run"],
)
def test_launch_returns_run_id_from_generated_launch_response(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    launch: Callable[[], str],
    error_cls: type[HirundoError],
) -> None:
    _patch_post(monkeypatch, module, _json_body(launch_response_payload(RUN_ID)))

    assert launch() == RUN_ID


@pytest.mark.parametrize(
    ("module", "launch", "error_cls"),
    LAUNCH_ENDPOINTS,
    ids=["unlearning-run", "behavior-eval-run"],
)
@pytest.mark.parametrize(
    "body",
    [
        _json_body({"message": "Run launched"}),
        _json_body(launch_response_payload("")),
        _json_body(launch_response_payload(None)),
        _json_body({"message": "Run launched", "hir_run_id": RUN_ID}),
        _json_body(RUN_ID),
        b"",
    ],
    ids=[
        "missing-run-id",
        "empty-run-id",
        "null-run-id",
        "unknown-run-id-key",
        "bare-run-id",
        "empty-body",
    ],
)
def test_launch_rejects_response_without_valid_run_id(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    launch: Callable[[], str],
    error_cls: type[HirundoError],
    body: bytes,
) -> None:
    _patch_post(monkeypatch, module, body)

    with pytest.raises(error_cls, match="invalid LaunchedRunResponse response"):
        launch()
