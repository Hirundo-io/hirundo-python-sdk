"""Tests for the opt-out SDK telemetry in `hirundo._telemetry`.

Every HTTP request goes through a fake `HTTPAdapter.send`, so the tests see the
exact requests the SDK would make. Real DNS lookups and outbound connections
fail the test, so no test here needs network access or credentials.
"""

import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import hirundo._env as sdk_env
import hirundo._headers as sdk_headers
import hirundo._telemetry as telemetry
import hirundo.unlearning_llm as unlearning_llm
import pytest
import requests
from hirundo.cli import typer_click_object
from hirundo.unlearning_llm import LlmRunInfo, LlmUnlearningRun, SecurityBehavior
from requests.adapters import HTTPAdapter

TEST_API_HOST = "https://api.hirundo.test"
TEST_API_KEY = "unit-test-api-key"
TEST_POSTHOG_HOST = "https://posthog.hirundo.test"
TEST_POSTHOG_KEY = "phc_unit_test_project_key"
TEST_USER_ID = 42
RUN_ID = "0b6f5d6c-run-id"
POSTHOG_ROOT_URL = f"{TEST_POSTHOG_HOST}/"
POSTHOG_CAPTURE_URL = f"{TEST_POSTHOG_HOST}/i/v0/e/"
API_KEY_LIST_URL = f"{TEST_API_HOST}/api-key/"
TELEMETRY_ENV_VARS = (
    "HIRUNDO_TELEMETRY_DISABLED",
    "DO_NOT_TRACK",
    "HIRUNDO_TELEMETRY_POSTHOG_KEY",
    "HIRUNDO_TELEMETRY_POSTHOG_HOST",
)
FLUSH_TIMEOUT_SECONDS = 5.0
COMMON_PROPERTY_NAMES = {
    "sdk_version",
    "python_version",
    "os",
    "os_version",
    "arch",
    "is_notebook",
    "is_hirundo_cloud",
    "$geoip_disable",
}
EVENT_FIELD_NAMES = {"api_key", "event", "distinct_id", "timestamp", "properties"}

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    body: JsonValue
    timeout: float | tuple[float, float] | None


@dataclass
class FakeResponse:
    status_code: int = 200
    body: JsonValue = None


Responder = Callable[[requests.PreparedRequest], FakeResponse]


@dataclass
class FakeTransport:
    """Answers every `requests` call made in the process."""

    routes: dict[tuple[str, str], Responder] = field(default_factory=dict)
    recorded: list[RecordedRequest] = field(default_factory=list)

    def answer(self, method: str, url: str, responder: Responder) -> None:
        self.routes[(method, url)] = responder

    def answer_json(
        self, method: str, url: str, body: JsonValue, status_code: int = 200
    ) -> None:
        self.answer(method, url, lambda _request: FakeResponse(status_code, body))

    def requests_to(self, method: str, url: str) -> list[RecordedRequest]:
        return [
            recorded
            for recorded in self.recorded
            if recorded.method == method and recorded.url == url
        ]

    def captured_events(self) -> list[dict[str, JsonValue]]:
        events: list[dict[str, JsonValue]] = []
        for recorded in self.requests_to("POST", POSTHOG_CAPTURE_URL):
            assert isinstance(recorded.body, dict)
            events.append(recorded.body)
        return events

    def send(
        self,
        request: requests.PreparedRequest,
        timeout: float | tuple[float, float] | None,
    ) -> requests.Response:
        method = request.method or ""
        url = request.url or ""
        raw_body = request.body
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode()
        body: JsonValue = json.loads(raw_body) if raw_body else None
        self.recorded.append(RecordedRequest(method, url, body, timeout))
        responder = self.routes.get((method, url))
        if responder is None:
            raise AssertionError(f"Unexpected request: {method} {url}")
        fake_response = responder(request)
        response = requests.Response()
        response.status_code = fake_response.status_code
        response._content = json.dumps(fake_response.body).encode()
        response.headers["Content-Type"] = "application/json"
        response.url = url
        response.request = request
        return response


def refuse_network(*_args: str | int, **_kwargs: str | int) -> None:
    raise AssertionError("Telemetry tests must not open real network connections")


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTransport]:
    fake_transport = FakeTransport()

    def fake_send(
        _adapter: HTTPAdapter,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float, float] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        return fake_transport.send(request, timeout)

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    yield fake_transport


@pytest.fixture
def telemetry_on(
    monkeypatch: pytest.MonkeyPatch, transport: FakeTransport
) -> FakeTransport:
    """Telemetry turned on, PostHog reachable and the API key owned by user 42."""
    for name in TELEMETRY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HIRUNDO_TELEMETRY_POSTHOG_KEY", TEST_POSTHOG_KEY)
    monkeypatch.setenv("HIRUNDO_TELEMETRY_POSTHOG_HOST", TEST_POSTHOG_HOST)
    monkeypatch.setattr(telemetry, "_STATE", telemetry._ReporterState())
    monkeypatch.setattr(telemetry, "API_HOST", TEST_API_HOST)
    monkeypatch.setattr(unlearning_llm, "API_HOST", TEST_API_HOST)
    monkeypatch.setattr(sdk_env, "API_KEY", TEST_API_KEY)
    monkeypatch.setattr(sdk_headers, "API_KEY", TEST_API_KEY)
    transport.answer_json("HEAD", POSTHOG_ROOT_URL, None)
    transport.answer_json(
        "GET",
        API_KEY_LIST_URL,
        [
            {
                "id": 1,
                "name": "laptop",
                "kind": "STANDARD",
                "user_owner_id": TEST_USER_ID,
            }
        ],
    )
    transport.answer_json("POST", POSTHOG_CAPTURE_URL, {"status": "Ok"})
    return transport


def flush_telemetry() -> None:
    telemetry._STATE.flush(FLUSH_TIMEOUT_SECONDS)


@telemetry.reports_errors(telemetry.Product.LLM_UNLEARNING)
def failing_sdk_call(error: Exception) -> None:
    raise error


@telemetry.reports_errors(telemetry.Product.LLM_UNLEARNING)
def working_sdk_call() -> str:
    return "done"


def http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError("server said no", response=response)


def use_sdk_while_telemetry_runs() -> None:
    with pytest.raises(ValueError):
        failing_sdk_call(ValueError("dataset my-private-dataset not found"))
    working_sdk_call()
    telemetry.report_run_launched(telemetry.Product.LLM_UNLEARNING, RUN_ID)
    telemetry.report_cli_command("check-run")


@pytest.mark.parametrize(
    "opt_out_env",
    [
        {"HIRUNDO_TELEMETRY_DISABLED": "true"},
        {"HIRUNDO_TELEMETRY_DISABLED": "1"},
        {"DO_NOT_TRACK": "1"},
        {"DO_NOT_TRACK": "true"},
        {"DO_NOT_TRACK": "not-a-boolean"},
        {"HIRUNDO_TELEMETRY_POSTHOG_KEY": ""},
    ],
)
def test_disabled_telemetry_makes_no_network_call_and_collects_nothing(
    monkeypatch: pytest.MonkeyPatch,
    telemetry_on: FakeTransport,
    opt_out_env: dict[str, str],
) -> None:
    # Opt-out must win over the built-in PostHog project key.
    monkeypatch.delenv("HIRUNDO_TELEMETRY_POSTHOG_KEY")
    for name, value in opt_out_env.items():
        monkeypatch.setenv(name, value)
    collected: list[str] = []
    monkeypatch.setattr(
        telemetry, "_common_properties", lambda: collected.append("properties")
    )
    threads_before = threading.active_count()

    use_sdk_while_telemetry_runs()
    flush_telemetry()

    assert telemetry._STATE.get() is None
    assert telemetry_on.recorded == []
    assert collected == []
    assert threading.active_count() == threads_before


def test_default_posthog_key_is_used_when_not_set(
    monkeypatch: pytest.MonkeyPatch, telemetry_on: FakeTransport
) -> None:
    monkeypatch.delenv("HIRUNDO_TELEMETRY_POSTHOG_KEY")

    use_sdk_while_telemetry_runs()
    flush_telemetry()

    events = telemetry_on.captured_events()
    assert events
    assert [event["api_key"] for event in events] == [
        telemetry.POSTHOG_PROJECT_KEY
    ] * len(events)


def test_posthog_key_setting_replaces_the_default(
    telemetry_on: FakeTransport,
) -> None:
    use_sdk_while_telemetry_runs()
    flush_telemetry()

    events = telemetry_on.captured_events()
    assert events
    assert [event["api_key"] for event in events] == [TEST_POSTHOG_KEY] * len(events)
    assert TEST_POSTHOG_KEY != telemetry.POSTHOG_PROJECT_KEY


def test_unreachable_posthog_means_no_events(telemetry_on: FakeTransport) -> None:
    def unreachable(_request: requests.PreparedRequest) -> FakeResponse:
        raise requests.ConnectionError("PostHog cannot be reached")

    telemetry_on.answer("HEAD", POSTHOG_ROOT_URL, unreachable)

    use_sdk_while_telemetry_runs()
    flush_telemetry()
    use_sdk_while_telemetry_runs()
    flush_telemetry()

    assert [(recorded.method, recorded.url) for recorded in telemetry_on.recorded] == [
        ("HEAD", POSTHOG_ROOT_URL)
    ]


@pytest.mark.parametrize(
    ("status_code", "api_keys"),
    [
        (401, {"detail": "Invalid API key"}),
        (200, []),
        (
            200,
            [
                {"kind": "STANDARD", "user_owner_id": 1},
                {"kind": "STANDARD", "user_owner_id": 2},
            ],
        ),
        (200, [{"kind": "CI", "user_owner_id": 1}]),
    ],
)
def test_unknown_user_id_means_no_events(
    telemetry_on: FakeTransport, status_code: int, api_keys: JsonValue
) -> None:
    telemetry_on.answer_json("GET", API_KEY_LIST_URL, api_keys, status_code)

    use_sdk_while_telemetry_runs()
    flush_telemetry()

    assert telemetry_on.captured_events() == []


def test_reachability_check_runs_once_per_process(
    telemetry_on: FakeTransport,
) -> None:
    for _ in range(3):
        use_sdk_while_telemetry_runs()
    reporter = telemetry._STATE.get()
    flush_telemetry()

    assert telemetry._STATE.get() is reporter
    assert len(telemetry_on.requests_to("HEAD", POSTHOG_ROOT_URL)) == 1
    assert len(telemetry_on.requests_to("GET", API_KEY_LIST_URL)) == 1
    assert len(telemetry_on.captured_events()) == 9
    assert {recorded.timeout for recorded in telemetry_on.recorded} == {
        telemetry.TELEMETRY_TIMEOUT_SECONDS
    }


def test_run_launch_event_contains_exactly_the_allowed_fields(
    telemetry_on: FakeTransport,
) -> None:
    telemetry_on.answer_json(
        "POST", f"{TEST_API_HOST}/unlearning-llm/run/7", {"run_id": RUN_ID}
    )

    run_id = LlmUnlearningRun.launch(
        model_id=7,
        run_info=LlmRunInfo(
            name="secret-run-name", target_behaviors=[SecurityBehavior()]
        ),
    )
    flush_telemetry()

    assert run_id == RUN_ID
    [event] = telemetry_on.captured_events()
    assert set(event) == EVENT_FIELD_NAMES
    assert event["api_key"] == TEST_POSTHOG_KEY
    assert event["event"] == "SDK run launched"
    assert event["distinct_id"] == str(TEST_USER_ID)
    properties = event["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == COMMON_PROPERTY_NAMES | {"product", "run_id"}
    assert properties["product"] == "llm_unlearning"
    assert properties["run_id"] == RUN_ID
    assert properties["$geoip_disable"] is True
    assert properties["is_hirundo_cloud"] is False
    serialized_event = json.dumps(event)
    for private_value in (TEST_API_KEY, TEST_API_HOST, "secret-run-name"):
        assert private_value not in serialized_event


def test_error_event_contains_exactly_the_allowed_fields(
    telemetry_on: FakeTransport,
) -> None:
    with pytest.raises(requests.HTTPError):
        failing_sdk_call(http_error(404))
    flush_telemetry()

    [event] = telemetry_on.captured_events()
    assert set(event) == EVENT_FIELD_NAMES
    assert event["event"] == "SDK error"
    properties = event["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == COMMON_PROPERTY_NAMES | {
        "product",
        "function",
        "error_type",
        "http_status",
    }
    assert properties["function"] == "failing_sdk_call"
    assert properties["error_type"] == "HTTPError"
    assert properties["http_status"] == 404
    assert "server said no" not in json.dumps(event)


def test_cli_command_event_contains_exactly_the_allowed_fields(
    telemetry_on: FakeTransport,
) -> None:
    with pytest.raises(SystemExit) as exited:
        typer_click_object.main(["set-api-key", "--help"], prog_name="hirundo")
    flush_telemetry()

    assert exited.value.code == 0
    [event] = telemetry_on.captured_events()
    assert set(event) == EVENT_FIELD_NAMES
    assert event["event"] == "SDK CLI command"
    properties = event["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == COMMON_PROPERTY_NAMES | {"command"}
    assert properties["command"] == "set-api-key"


@pytest.mark.parametrize(
    "error", [ValueError("bad input"), http_error(500), KeyError("run_id")]
)
def test_errors_are_reraised_unchanged(
    telemetry_on: FakeTransport, error: Exception
) -> None:
    with pytest.raises(type(error)) as raised:
        failing_sdk_call(error)
    flush_telemetry()

    assert raised.value is error
    assert raised.traceback[-1].name == "failing_sdk_call"
    assert len(telemetry_on.captured_events()) == 1


def test_failing_launch_reraises_the_sdk_error_unchanged(
    telemetry_on: FakeTransport,
) -> None:
    telemetry_on.answer_json(
        "POST",
        f"{TEST_API_HOST}/unlearning-llm/run/7",
        {"detail": "Model not found"},
        status_code=404,
    )

    with pytest.raises(requests.HTTPError) as raised:
        LlmUnlearningRun.launch(
            model_id=7, run_info=LlmRunInfo(target_behaviors=[SecurityBehavior()])
        )
    flush_telemetry()

    assert raised.value.response is not None
    assert raised.value.response.status_code == 404
    [event] = telemetry_on.captured_events()
    properties = event["properties"]
    assert isinstance(properties, dict)
    assert properties["function"] == "LlmUnlearningRun.launch"
    assert properties["http_status"] == 404


@pytest.mark.parametrize("status_code", [400, 401, 500, 503])
def test_rejected_capture_is_logged_and_dropped(
    telemetry_on: FakeTransport,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
) -> None:
    telemetry_on.answer_json(
        "POST", POSTHOG_CAPTURE_URL, {"detail": "rejected"}, status_code
    )
    caplog.set_level(logging.DEBUG, logger=telemetry.__name__)

    use_sdk_while_telemetry_runs()
    flush_telemetry()

    assert len(telemetry_on.captured_events()) == 3
    rejection_logs = [
        record
        for record in caplog.records
        if record.name == telemetry.__name__
        and record.levelno == logging.DEBUG
        and f"HTTP {status_code}" in record.getMessage()
    ]
    assert len(rejection_logs) == 3
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_hanging_posthog_does_not_slow_sdk_calls(
    telemetry_on: FakeTransport,
) -> None:
    release_posthog = threading.Event()

    def hanging(_request: requests.PreparedRequest) -> FakeResponse:
        release_posthog.wait(FLUSH_TIMEOUT_SECONDS)
        return FakeResponse()

    telemetry_on.answer("HEAD", POSTHOG_ROOT_URL, hanging)
    telemetry_on.answer("POST", POSTHOG_CAPTURE_URL, hanging)

    started_at = time.perf_counter()
    for _ in range(10):
        use_sdk_while_telemetry_runs()
    elapsed_seconds = time.perf_counter() - started_at

    release_posthog.set()
    flush_telemetry()
    assert elapsed_seconds < 0.5
