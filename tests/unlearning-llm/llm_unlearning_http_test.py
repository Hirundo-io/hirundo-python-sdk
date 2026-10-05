"""Mocked HTTP tests for the LLM unlearning client in `hirundo.unlearning_llm`.

The REST calls go through the SDK's real retrying `requests` session. A fake
transport adapter is mounted on that session, so the tests check the request
that `requests` actually prepares: method, URL with query string, headers, JSON
body and timeout. Run status checks go through real `httpx` clients backed by
`httpx.MockTransport`, so `httpx-sse` parses a real `text/event-stream` body.

No test here needs network access, an API key, or cloud credentials. Real DNS
lookups and outbound connections fail the test.
"""

import json
import socket
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import hirundo._env as sdk_env
import hirundo._headers as sdk_headers
import hirundo._http as sdk_http
import hirundo._run_checking as run_checking
import hirundo.unlearning_llm as unlearning_llm
import httpx
import pytest
import requests
from hirundo import HirundoError
from hirundo._llm_sources import HuggingFaceTransformersModel, LocalTransformersModel
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT, SSE_TIMEOUT
from hirundo.unlearning_llm import (
    BiasBehavior,
    LlmModel,
    LlmModelOut,
    LlmRunInfo,
    LlmUnlearningRun,
    OutputUnlearningLlmRun,
    SecurityBehavior,
)
from requests.adapters import BaseAdapter

TEST_API_HOST = "https://api.hirundo.test"
TEST_API_KEY = "unit-test-api-key"
EXPECTED_JSON_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {TEST_API_KEY}",
    "HIRUNDO-API-VERSION": "0.3",
}
RUN_ID = "0b6f5d6c-run-id"


# --------------------------------------------------------------------------- #
# Fake REST API (requests transport adapter)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout: object

    def json_body(self) -> Any:
        assert self.body is not None, "Request had no body"
        return json.loads(self.body)


@dataclass(frozen=True)
class CannedResponse:
    status_code: int
    payload: object = None
    raw_body: bytes | None = None


class FakeHirundoApi(BaseAdapter):
    """Transport adapter that serves canned responses for exact method + URL."""

    def __init__(self) -> None:
        super().__init__()
        self.routes: dict[tuple[str, str], CannedResponse] = {}
        self.requests: list[RecordedRequest] = []

    def add(
        self,
        method: str,
        path: str,
        *,
        status_code: int = 200,
        payload: object = None,
        raw_body: bytes | None = None,
    ) -> None:
        self.routes[(method, f"{TEST_API_HOST}{path}")] = CannedResponse(
            status_code=status_code, payload=payload, raw_body=raw_body
        )

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: object = None,
        verify: bool | str = True,
        cert: object = None,
        proxies: object = None,
    ) -> requests.Response:
        method = request.method or ""
        url = request.url or ""
        raw_request_body = request.body
        request_body = (
            raw_request_body.encode()
            if isinstance(raw_request_body, str)
            else raw_request_body
        )
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=dict(request.headers),
                body=request_body if isinstance(request_body, bytes) else None,
                timeout=timeout,
            )
        )
        canned_response = self.routes.get((method, url))
        if canned_response is None:
            raise AssertionError(f"Unexpected request: {method} {url}")

        response = requests.Response()
        response.status_code = canned_response.status_code
        response.reason = "OK" if canned_response.status_code < 400 else "Error"
        response.url = url
        response.request = request
        if canned_response.raw_body is not None:
            response._content = canned_response.raw_body
        else:
            response._content = json.dumps(canned_response.payload).encode()
            response.headers["Content-Type"] = "application/json"
        return response

    def close(self) -> None:
        return None

    def only_request(self) -> RecordedRequest:
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


# --------------------------------------------------------------------------- #
# Fake SSE endpoint (httpx MockTransport)
# --------------------------------------------------------------------------- #


def _sse_stream(*events: dict[str, object] | None) -> bytes:
    """Encode events as an SSE body. `None` encodes a keep-alive ping."""
    lines: list[str] = []
    for event_index, event in enumerate(events):
        if event is None:
            lines.append("event: ping\ndata: {}\n\n")
        else:
            lines.append(f"id: {event_index}\ndata: {json.dumps(event)}\n\n")
    return "".join(lines).encode()


def _run_event(state: str | None, result: object = None) -> dict[str, object]:
    event_data: dict[str, object] = {"id": RUN_ID, "result": result}
    if state is not None:
        event_data["state"] = state
    return {"data": event_data}


@dataclass
class FakeSseEndpoint:
    """Serves one queued SSE body per connection to the run status endpoint."""

    streams: list[bytes] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    client_timeouts: list[object] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        expected_url = f"{TEST_API_HOST}/unlearning-llm/run/{RUN_ID}"
        if request.method != "GET" or str(request.url) != expected_url:
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")
        if not self.streams:
            raise AssertionError("SSE endpoint was called more times than expected")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=self.streams.pop(0),
        )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Real network access attempted: {args!r}")

    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)


@pytest.fixture(autouse=True)
def _configure_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unlearning_llm, "API_HOST", TEST_API_HOST)
    monkeypatch.setattr(sdk_env, "API_KEY", TEST_API_KEY)
    monkeypatch.setattr(sdk_headers, "API_KEY", TEST_API_KEY)


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeHirundoApi:
    fake_adapter = FakeHirundoApi()
    # Mount ahead of the real adapters, as `Session.mount` would for a longer prefix.
    adapters: OrderedDict[str, BaseAdapter] = OrderedDict(
        [(f"{TEST_API_HOST}/", fake_adapter)]
    )
    adapters.update(sdk_http._SESSION.adapters)
    monkeypatch.setattr(sdk_http._SESSION, "adapters", adapters)
    return fake_adapter


@pytest.fixture
def fake_sse(monkeypatch: pytest.MonkeyPatch) -> FakeSseEndpoint:
    sse_endpoint = FakeSseEndpoint()
    transport = httpx.MockTransport(sse_endpoint.handle)
    original_client = httpx.Client
    original_async_client = httpx.AsyncClient

    def build_client(**client_kwargs: Any) -> httpx.Client:
        sse_endpoint.client_timeouts.append(client_kwargs.get("timeout"))
        return original_client(transport=transport, **client_kwargs)

    def build_async_client(**client_kwargs: Any) -> httpx.AsyncClient:
        sse_endpoint.client_timeouts.append(client_kwargs.get("timeout"))
        return original_async_client(transport=transport, **client_kwargs)

    monkeypatch.setattr(run_checking.httpx, "Client", build_client)
    monkeypatch.setattr(run_checking.httpx, "AsyncClient", build_async_client)
    return sse_endpoint


# --------------------------------------------------------------------------- #
# Payload builders
# --------------------------------------------------------------------------- #


def _llm_model_out_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 7,
        "organization_id": 3,
        "creator_id": 11,
        "creator_name": "Unit Test",
        "created_at": "2026-10-01T12:00:00Z",
        "updated_at": "2026-10-02T12:00:00Z",
        "model_name": "unit-test-llm",
        "model_source": {
            "type": "huggingface_transformers",
            "model_name": "org/base-model",
        },
    }
    payload.update(overrides)
    return payload


def _unlearning_run_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 21,
        "name": "unit-test-run",
        "model_id": 7,
        "model": {"id": 7, "model_name": "unit-test-llm"},
        "target_behaviors": [{"type": "BIAS", "bias_type": "ALL"}],
        "target_utilities": [],
        "advanced_options": None,
        "aggressiveness": 0.5,
        "run_id": RUN_ID,
        "mlflow_run_id": None,
        "status": "SUCCESS",
        "approved": True,
        "created_at": "2026-10-01T12:00:00Z",
        "completed_at": "2026-10-01T13:00:00Z",
        "pre_process_progress": 100.0,
        "optimization_progress": 100.0,
        "post_process_progress": 100.0,
    }
    payload.update(overrides)
    return payload


def _created_llm_model() -> LlmModel:
    return LlmModel(
        id=7,
        organization_id=3,
        model_name="unit-test-llm",
        model_source=HuggingFaceTransformersModel(model_name="org/base-model"),
    )


# --------------------------------------------------------------------------- #
# LlmModel
# --------------------------------------------------------------------------- #


def test_llm_model_create_posts_model_and_stores_id(fake_api: FakeHirundoApi) -> None:
    fake_api.add("POST", "/unlearning-llm/llm/", payload={"id": 42})
    llm_model = LlmModel(
        model_name="unit-test-llm",
        model_source=HuggingFaceTransformersModel(
            model_name="org/base-model", revision="main"
        ),
    )

    created_id = llm_model.create(replace_if_exists=True)

    assert created_id == 42
    assert llm_model.id == 42
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "POST"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/llm/"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == MODIFY_TIMEOUT
    assert recorded_request.json_body() == {
        "id": None,
        "organization_id": None,
        "model_name": "unit-test-llm",
        "model_source": {
            "type": "huggingface_transformers",
            "revision": "main",
            "code_revision": None,
            "model_name": "org/base-model",
            "token": None,
        },
        "archive_existing_runs": True,
        "replace_if_exists": True,
    }


def test_llm_model_get_by_id_parses_model(fake_api: FakeHirundoApi) -> None:
    fake_api.add("GET", "/unlearning-llm/llm/7", payload=_llm_model_out_payload())

    llm_model = LlmModel.get_by_id(7)

    assert isinstance(llm_model, LlmModelOut)
    assert llm_model.id == 7
    assert llm_model.organization_id == 3
    assert llm_model.model_name == "unit-test-llm"
    assert llm_model.model_source.model_dump(mode="json") == {
        "type": "huggingface_transformers",
        "model_name": "org/base-model",
    }
    assert llm_model.created_at.isoformat() == "2026-10-01T12:00:00+00:00"
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "GET"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/llm/7"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.body is None
    assert recorded_request.timeout == READ_TIMEOUT


def test_llm_model_get_by_name_parses_local_model(fake_api: FakeHirundoApi) -> None:
    fake_api.add(
        "GET",
        "/unlearning-llm/llm/by-name/unit-test-llm",
        payload=_llm_model_out_payload(
            model_source={"type": "local_transformers", "local_path": "/models/llm"}
        ),
    )

    llm_model = LlmModel.get_by_name("unit-test-llm")

    assert isinstance(llm_model.model_source, LocalTransformersModel)
    assert llm_model.model_source.local_path == "/models/llm"
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "GET"
    assert (
        recorded_request.url
        == f"{TEST_API_HOST}/unlearning-llm/llm/by-name/unit-test-llm"
    )
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == READ_TIMEOUT


@pytest.mark.parametrize(
    ("organization_id", "expected_path"),
    [
        (None, "/unlearning-llm/llm/"),
        (3, "/unlearning-llm/llm/?model_organization_id=3"),
    ],
)
def test_llm_model_list_sends_organization_filter(
    fake_api: FakeHirundoApi,
    organization_id: int | None,
    expected_path: str,
) -> None:
    fake_api.add(
        "GET",
        expected_path,
        payload=[
            _llm_model_out_payload(id=7),
            _llm_model_out_payload(id=8, model_name="second-llm"),
        ],
    )

    llm_models = LlmModel.list(organization_id=organization_id)

    assert [llm_model.id for llm_model in llm_models] == [7, 8]
    assert [llm_model.model_name for llm_model in llm_models] == [
        "unit-test-llm",
        "second-llm",
    ]
    assert all(isinstance(llm_model, LlmModelOut) for llm_model in llm_models)
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "GET"
    assert recorded_request.url == f"{TEST_API_HOST}{expected_path}"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == READ_TIMEOUT


def test_llm_model_update_puts_changes_and_updates_local_state(
    fake_api: FakeHirundoApi,
) -> None:
    fake_api.add("PUT", "/unlearning-llm/llm/7", payload=_llm_model_out_payload())
    llm_model = _created_llm_model()
    new_source = LocalTransformersModel(local_path="/models/renamed")

    update_result = llm_model.update(
        model_name="renamed-llm",
        model_source=new_source,
        archive_existing_runs=False,
    )

    assert update_result is None
    assert llm_model.model_name == "renamed-llm"
    assert llm_model.model_source == new_source
    assert llm_model.archive_existing_runs is False
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "PUT"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/llm/7"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == MODIFY_TIMEOUT
    assert recorded_request.json_body() == {
        "model_name": "renamed-llm",
        "model_source": {
            "type": "local_transformers",
            "revision": None,
            "code_revision": None,
            "local_path": "/models/renamed",
        },
        "archive_existing_runs": False,
        "organization_id": 3,
    }


def test_llm_model_partial_update_sends_nulls_for_unchanged_fields(
    fake_api: FakeHirundoApi,
) -> None:
    fake_api.add("PUT", "/unlearning-llm/llm/7", payload=_llm_model_out_payload())
    llm_model = _created_llm_model()
    original_source = llm_model.model_source

    llm_model.update(model_name="renamed-llm")

    assert llm_model.model_name == "renamed-llm"
    assert llm_model.model_source == original_source
    assert llm_model.archive_existing_runs is True
    assert fake_api.only_request().json_body() == {
        "model_name": "renamed-llm",
        "model_source": None,
        "archive_existing_runs": None,
        "organization_id": 3,
    }


def test_llm_model_delete_sends_delete(fake_api: FakeHirundoApi) -> None:
    fake_api.add("DELETE", "/unlearning-llm/llm/7", payload=None)

    delete_result = _created_llm_model().delete()

    assert delete_result is None
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "DELETE"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/llm/7"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.body is None
    assert recorded_request.timeout == MODIFY_TIMEOUT


def test_llm_model_delete_by_id_sends_delete(fake_api: FakeHirundoApi) -> None:
    fake_api.add("DELETE", "/unlearning-llm/llm/9", payload=None)

    LlmModel.delete_by_id(9)

    recorded_request = fake_api.only_request()
    assert recorded_request.method == "DELETE"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/llm/9"


@pytest.mark.parametrize(
    "call_uncreated_model",
    [
        lambda llm_model: llm_model.update(model_name="renamed-llm"),
        lambda llm_model: llm_model.delete(),
    ],
    ids=["update", "delete"],
)
def test_llm_model_mutations_require_created_model(
    fake_api: FakeHirundoApi,
    call_uncreated_model: Callable[[LlmModel], None],
) -> None:
    llm_model = LlmModel(
        model_name="unit-test-llm",
        model_source=HuggingFaceTransformersModel(model_name="org/base-model"),
    )

    with pytest.raises(ValueError, match="No LLM model has been created"):
        call_uncreated_model(llm_model)

    assert fake_api.requests == []


# --------------------------------------------------------------------------- #
# LlmUnlearningRun REST calls
# --------------------------------------------------------------------------- #


def test_launch_posts_run_info_and_returns_run_id(fake_api: FakeHirundoApi) -> None:
    fake_api.add("POST", "/unlearning-llm/run/7", payload={"run_id": RUN_ID})
    run_info = LlmRunInfo(
        name="unit-test-run",
        target_behaviors=[BiasBehavior(), SecurityBehavior()],
        aggressiveness=0.25,
    )

    run_id = LlmUnlearningRun.launch(model_id=7, run_info=run_info)

    assert run_id == RUN_ID
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "POST"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/run/7"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == MODIFY_TIMEOUT
    assert recorded_request.json_body() == {
        "organization_id": None,
        "name": "unit-test-run",
        "target_behaviors": [
            {"type": "BIAS", "bias_type": "ALL"},
            {"type": "SECURITY"},
        ],
        "target_utilities": [],
        "advanced_options": None,
        "aggressiveness": 0.25,
    }


def test_launch_accepts_bare_run_id_response(fake_api: FakeHirundoApi) -> None:
    fake_api.add("POST", "/unlearning-llm/run/7", payload=RUN_ID)

    run_id = LlmUnlearningRun.launch(
        model_id=7, run_info=LlmRunInfo(target_behaviors=[SecurityBehavior()])
    )

    assert run_id == RUN_ID
    assert "aggressiveness" not in fake_api.only_request().json_body()


@pytest.mark.parametrize(
    "response_kwargs",
    [{"payload": {"status": "queued"}}, {"raw_body": b""}],
    ids=["json-without-run-id", "empty-body"],
)
def test_launch_rejects_response_without_run_id(
    fake_api: FakeHirundoApi,
    response_kwargs: dict[str, Any],
) -> None:
    fake_api.add("POST", "/unlearning-llm/run/7", **response_kwargs)

    with pytest.raises(ValueError, match="No run ID returned from launch request"):
        LlmUnlearningRun.launch(
            model_id=7, run_info=LlmRunInfo(target_behaviors=[SecurityBehavior()])
        )


@pytest.mark.parametrize(
    ("call_run_action", "expected_path"),
    [
        (lambda: LlmUnlearningRun.cancel(RUN_ID), f"/run/cancel/{RUN_ID}"),
        (lambda: LlmUnlearningRun.archive(RUN_ID), f"/run/archive/{RUN_ID}"),
        (lambda: LlmUnlearningRun.restore(RUN_ID), f"/run/restore/{RUN_ID}"),
    ],
    ids=["cancel", "archive", "restore"],
)
def test_run_state_actions_send_bodyless_patch(
    fake_api: FakeHirundoApi,
    call_run_action: Callable[[], None],
    expected_path: str,
) -> None:
    fake_api.add("PATCH", f"/unlearning-llm{expected_path}", payload=None)

    action_result = call_run_action()

    assert action_result is None
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "PATCH"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm{expected_path}"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.body is None
    assert recorded_request.timeout == MODIFY_TIMEOUT


def test_rename_patches_new_name(fake_api: FakeHirundoApi) -> None:
    fake_api.add("PATCH", f"/unlearning-llm/run/rename/{RUN_ID}", payload=None)

    rename_result = LlmUnlearningRun.rename(RUN_ID, "renamed-run")

    assert rename_result is None
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "PATCH"
    assert recorded_request.url == f"{TEST_API_HOST}/unlearning-llm/run/rename/{RUN_ID}"
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == MODIFY_TIMEOUT
    assert recorded_request.json_body() == {"new_name": "renamed-run"}


@pytest.mark.parametrize(
    ("list_kwargs", "expected_query"),
    [
        ({}, "archived=False"),
        (
            {"organization_id": 3, "archived": True},
            "archived=True&unlearning_organization_id=3",
        ),
    ],
    ids=["defaults", "archived-for-organization"],
)
def test_run_list_sends_filters_and_parses_runs(
    fake_api: FakeHirundoApi,
    list_kwargs: dict[str, Any],
    expected_query: str,
) -> None:
    fake_api.add(
        "GET",
        f"/unlearning-llm/run/list?{expected_query}",
        payload=[
            _unlearning_run_payload(),
            _unlearning_run_payload(
                id=22,
                run_id="second-run-id",
                status="FAILURE",
                target_behaviors=[{"type": "SECURITY"}],
                aggressiveness=None,
                completed_at=None,
            ),
        ],
    )

    runs = LlmUnlearningRun.list(**list_kwargs)

    assert all(isinstance(run, OutputUnlearningLlmRun) for run in runs)
    assert [run.run_id for run in runs] == [RUN_ID, "second-run-id"]
    assert [run.status for run in runs] == ["SUCCESS", "FAILURE"]
    assert runs[0].target_behaviors[0].model_dump(mode="json") == {
        "type": "BIAS",
        "bias_type": "ALL",
    }
    assert runs[1].completed_at is None
    recorded_request = fake_api.only_request()
    assert recorded_request.method == "GET"
    assert (
        recorded_request.url
        == f"{TEST_API_HOST}/unlearning-llm/run/list?{expected_query}"
    )
    assert recorded_request.headers.items() >= EXPECTED_JSON_HEADERS.items()
    assert recorded_request.timeout == READ_TIMEOUT


def test_run_list_wraps_single_run_response(fake_api: FakeHirundoApi) -> None:
    fake_api.add(
        "GET",
        "/unlearning-llm/run/list?archived=False",
        payload=_unlearning_run_payload(),
    )

    runs = LlmUnlearningRun.list()

    assert [run.id for run in runs] == [21]


# --------------------------------------------------------------------------- #
# HTTP error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("method", "path", "call_client", "status_code", "error_payload", "message"),
    [
        (
            "GET",
            "/unlearning-llm/llm/7",
            lambda: LlmModel.get_by_id(7),
            404,
            {"detail": "LLM not found"},
            "404 Client Error: LLM not found",
        ),
        (
            "POST",
            "/unlearning-llm/llm/",
            lambda: LlmModel(
                model_name="unit-test-llm",
                model_source=HuggingFaceTransformersModel(model_name="org/model"),
            ).create(),
            409,
            {"reason": "Model name already exists"},
            "409 Client Error: Model name already exists",
        ),
        (
            "POST",
            "/unlearning-llm/run/7",
            lambda: LlmUnlearningRun.launch(
                7, LlmRunInfo(target_behaviors=[SecurityBehavior()])
            ),
            500,
            {"detail": "Worker pool unavailable"},
            "500 Server Error: Worker pool unavailable",
        ),
        (
            "PATCH",
            f"/unlearning-llm/run/cancel/{RUN_ID}",
            lambda: LlmUnlearningRun.cancel(RUN_ID),
            403,
            {"detail": "Forbidden"},
            "403 Client Error: Forbidden",
        ),
        (
            "GET",
            "/unlearning-llm/run/list?archived=False",
            lambda: LlmUnlearningRun.list(),
            503,
            "not a JSON object",
            "503 Server Error: Error",
        ),
    ],
    ids=[
        "get-by-id-404",
        "create-409",
        "launch-500",
        "cancel-403",
        "list-503-non-dict",
    ],
)
def test_http_errors_raise_with_server_reason(
    fake_api: FakeHirundoApi,
    method: str,
    path: str,
    call_client: Callable[[], object],
    status_code: int,
    error_payload: object,
    message: str,
) -> None:
    fake_api.add(method, path, status_code=status_code, payload=error_payload)

    with pytest.raises(requests.HTTPError, match=message) as raised_error:
        call_client()

    assert raised_error.value.response is not None
    assert raised_error.value.response.status_code == status_code
    assert len(fake_api.requests) == 1


# --------------------------------------------------------------------------- #
# check_run (sync SSE)
# --------------------------------------------------------------------------- #


def _assert_sse_requests(sse_endpoint: FakeSseEndpoint, expected_count: int) -> None:
    assert len(sse_endpoint.requests) == expected_count
    for sse_request in sse_endpoint.requests:
        assert sse_request.headers["Accept"] == "text/event-stream"
        assert sse_request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
        assert sse_request.headers["HIRUNDO-API-VERSION"] == "0.3"
    assert sse_endpoint.client_timeouts == [SSE_TIMEOUT] * expected_count


def test_check_run_returns_result_on_success(fake_sse: FakeSseEndpoint) -> None:
    success_result = {"model_id": 7, "adapter_url": "s3://bucket/adapter.zip"}
    fake_sse.streams.append(
        _sse_stream(
            None,
            _run_event("PENDING"),
            _run_event("STARTED"),
            _run_event(None, "Optimizing: 40.0% done"),
            _run_event(None, {"result": "Optimizing: 100.0% done"}),
            _run_event("SUCCESS", success_result),
        )
    )

    run_result = LlmUnlearningRun.check_run(RUN_ID)

    assert run_result == success_result
    _assert_sse_requests(fake_sse, expected_count=1)


def test_check_run_returns_whole_event_when_success_has_no_result(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.append(_sse_stream(_run_event("SUCCESS")))

    run_result = LlmUnlearningRun.check_run(RUN_ID)

    assert run_result == {"id": RUN_ID, "result": None, "state": "SUCCESS"}


def test_check_run_reconnects_while_run_is_pending(fake_sse: FakeSseEndpoint) -> None:
    fake_sse.streams.extend(
        [
            _sse_stream(_run_event("PENDING")),
            _sse_stream(_run_event("STARTED"), _run_event("SUCCESS", {"done": True})),
        ]
    )

    run_result = LlmUnlearningRun.check_run(RUN_ID)

    assert run_result == {"done": True}
    _assert_sse_requests(fake_sse, expected_count=2)


@pytest.mark.parametrize(
    ("terminal_state", "result", "message"),
    [
        (
            "FAILURE",
            "CUDA out of memory",
            "LLM unlearning run failed with error: CUDA out of memory",
        ),
        ("REJECTED", None, "LLM unlearning run failed with an unknown error"),
        ("REVOKED", None, "LLM unlearning run failed with an unknown error"),
    ],
)
def test_check_run_raises_on_terminal_failure_states(
    fake_sse: FakeSseEndpoint,
    terminal_state: str,
    result: str | None,
    message: str,
) -> None:
    fake_sse.streams.append(
        _sse_stream(_run_event("STARTED"), _run_event(terminal_state, result))
    )

    with pytest.raises(HirundoError, match=message):
        LlmUnlearningRun.check_run(RUN_ID)

    _assert_sse_requests(fake_sse, expected_count=1)


def test_check_run_raises_on_error_event(fake_sse: FakeSseEndpoint) -> None:
    fake_sse.streams.append(_sse_stream({"detail": "Run not found"}))

    with pytest.raises(HirundoError, match="Run not found"):
        LlmUnlearningRun.check_run(RUN_ID)


def test_check_run_stops_on_manual_approval_when_requested(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.append(
        _sse_stream(_run_event("STARTED"), _run_event("AWAITING MANUAL APPROVAL"))
    )

    run_result = LlmUnlearningRun.check_run(RUN_ID, stop_on_manual_approval=True)

    assert run_result is None


def test_check_run_raises_when_stream_ends_without_terminal_state(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.append(_sse_stream(_run_event("STARTED")))

    with pytest.raises(
        HirundoError, match="LLM unlearning run failed with an unknown error"
    ):
        LlmUnlearningRun.check_run(RUN_ID)


# --------------------------------------------------------------------------- #
# acheck_run (async SSE)
# --------------------------------------------------------------------------- #


async def _collect_async_events() -> list[dict]:
    return [event async for event in LlmUnlearningRun.acheck_run(RUN_ID)]


@pytest.mark.asyncio
async def test_acheck_run_yields_events_until_success(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.append(
        _sse_stream(
            None,
            _run_event("PENDING"),
            _run_event("STARTED", "Optimizing: 40.0% done"),
            _run_event("SUCCESS", {"done": True}),
        )
    )

    events = await _collect_async_events()

    assert events == [
        {"id": RUN_ID, "result": None, "state": "PENDING"},
        {"id": RUN_ID, "result": "Optimizing: 40.0% done", "state": "STARTED"},
        {"id": RUN_ID, "result": {"done": True}, "state": "SUCCESS"},
    ]
    _assert_sse_requests(fake_sse, expected_count=1)


@pytest.mark.asyncio
async def test_acheck_run_reconnects_while_run_is_pending(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.extend(
        [
            _sse_stream(_run_event("PENDING")),
            _sse_stream(_run_event("SUCCESS", {"done": True})),
        ]
    )

    events = await _collect_async_events()

    assert [event["state"] for event in events] == ["PENDING", "SUCCESS"]
    _assert_sse_requests(fake_sse, expected_count=2)


@pytest.mark.asyncio
async def test_acheck_run_yields_failure_event_and_stops(
    fake_sse: FakeSseEndpoint,
) -> None:
    fake_sse.streams.append(
        _sse_stream(_run_event("STARTED"), _run_event("FAILURE", "CUDA out of memory"))
    )

    events = await _collect_async_events()

    assert events[-1] == {
        "id": RUN_ID,
        "result": "CUDA out of memory",
        "state": "FAILURE",
    }
    _assert_sse_requests(fake_sse, expected_count=1)


@pytest.mark.asyncio
async def test_acheck_run_raises_on_error_event(fake_sse: FakeSseEndpoint) -> None:
    fake_sse.streams.append(_sse_stream({"reason": "Run not found"}))

    with pytest.raises(HirundoError, match="Run not found"):
        await _collect_async_events()
