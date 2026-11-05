"""Opt-out usage and error telemetry for the Hirundo SDK.

The README "Telemetry" section lists everything that is sent and how to turn it
off. Telemetry must never slow down or change an SDK call: every network request
happens on one background thread with a short timeout and no retries, and any
failure is logged at DEBUG level and dropped.
"""

import atexit
import functools
import os
import platform
import queue
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple, ParamSpec, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, ValidationError

import hirundo.logger
from hirundo._env import API_HOST, HIRUNDO_CLOUD_API_HOST
from hirundo._headers import get_headers

logger = hirundo.logger.get_logger(__name__)

POSTHOG_US_HOST = "https://us.i.posthog.com"
POSTHOG_CAPTURE_PATH = "/i/v0/e/"
TELEMETRY_TIMEOUT_SECONDS = 2.0
MAX_PENDING_EVENTS = 100
STANDARD_API_KEY_KIND = "STANDARD"

RUN_LAUNCHED_EVENT = "SDK run launched"
ERROR_EVENT = "SDK error"
CLI_COMMAND_EVENT = "SDK CLI command"

TELEMETRY_ON_MESSAGE = (
    "Hirundo SDK telemetry is on. It sends your Hirundo user ID, run IDs, SDK "
    "commands and error types to Hirundo. Set HIRUNDO_TELEMETRY_DISABLED=true "
    "or DO_NOT_TRACK=1 to turn it off."
)

FunctionParams = ParamSpec("FunctionParams")
ReturnT = TypeVar("ReturnT")


class Product(str, Enum):
    DATASET_QA = "dataset_qa"
    LLM_UNLEARNING = "llm_unlearning"
    LLM_BEHAVIOR_EVAL = "llm_behavior_eval"
    EXTERNAL_EVAL = "external_eval"


class TelemetrySettings(BaseModel):
    """Telemetry settings, read from the environment, `.env` or `~/.hirundo.conf`."""

    model_config = ConfigDict(frozen=True)

    disabled: bool = Field(default=False, validation_alias="HIRUNDO_TELEMETRY_DISABLED")
    do_not_track: bool = Field(default=False, validation_alias="DO_NOT_TRACK")
    posthog_project_key: str | None = Field(
        default=None, min_length=1, validation_alias="HIRUNDO_TELEMETRY_POSTHOG_KEY"
    )
    posthog_host: HttpUrl = Field(
        default=HttpUrl(POSTHOG_US_HOST),
        validation_alias="HIRUNDO_TELEMETRY_POSTHOG_HOST",
    )

    @classmethod
    def from_environment(cls) -> "TelemetrySettings | None":
        """Return the settings, or `None` when they are invalid (telemetry off)."""
        try:
            return cls.model_validate(dict(os.environ))
        except ValidationError as error:
            logger.debug(
                "Invalid telemetry settings, so telemetry is off: %s",
                [str(detail["loc"]) for detail in error.errors()],
            )
            return None

    @property
    def opted_out(self) -> bool:
        return self.disabled or self.do_not_track

    @property
    def posthog_root(self) -> str:
        return str(self.posthog_host).rstrip("/")


class CommonProperties(BaseModel):
    """Properties sent with every event."""

    sdk_version: str
    python_version: str
    os: str
    os_version: str
    arch: str
    is_notebook: bool
    is_hirundo_cloud: bool


class RunLaunchedProperties(BaseModel):
    product: Product
    run_id: str


class ErrorProperties(BaseModel):
    product: Product
    function: str
    error_type: str
    http_status: int | None


class CliCommandProperties(BaseModel):
    command: str


class _ApiKeyOwner(BaseModel):
    kind: str
    user_owner_id: int


_API_KEY_OWNERS = TypeAdapter(list[_ApiKeyOwner])


class _PendingEvent(NamedTuple):
    name: str
    properties: BaseModel
    timestamp: datetime


def _common_properties() -> CommonProperties:
    from hirundo import __version__

    return CommonProperties(
        sdk_version=__version__,
        python_version=platform.python_version(),
        os=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        is_notebook="ipykernel" in sys.modules,
        is_hirundo_cloud=(
            API_HOST is not None and API_HOST.rstrip("/") == HIRUNDO_CLOUD_API_HOST
        ),
    )


class _Reporter:
    """Sends events from one background thread.

    The thread first checks once that PostHog can be reached and looks up the
    Hirundo user ID. If either fails, it stops and every event is dropped.
    """

    def __init__(self, settings: TelemetrySettings, project_key: str) -> None:
        self._settings = settings
        self._project_key = project_key
        self._session = requests.Session()
        self._pending: queue.Queue[_PendingEvent | None] = queue.Queue(
            maxsize=MAX_PENDING_EVENTS
        )
        self._stopped = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="hirundo-telemetry", daemon=True
        )
        self._worker.start()

    def submit(self, event: _PendingEvent) -> None:
        if self._stopped.is_set():
            return
        try:
            self._pending.put_nowait(event)
        except queue.Full:
            logger.debug("Telemetry queue is full, so the event was dropped")

    def flush(self, timeout_seconds: float) -> None:
        """Send queued events, waiting at most `timeout_seconds`, then stop."""
        try:
            self._pending.put_nowait(None)
        except queue.Full:
            logger.debug("Telemetry queue is full, so pending events are dropped")
        self._worker.join(timeout_seconds)

    def _run(self) -> None:
        try:
            user_id = self._connect()
            if user_id is None:
                return
            logger.info(TELEMETRY_ON_MESSAGE)
            common_properties = _common_properties()
            while (event := self._pending.get()) is not None:
                self._send(event, user_id, common_properties)
        except Exception as error:
            logger.debug("Telemetry stopped: %s", type(error).__name__)
        finally:
            self._stopped.set()

    def _connect(self) -> str | None:
        """Check once that PostHog can be reached, then return the user ID."""
        try:
            self._session.head(
                self._settings.posthog_root,
                timeout=TELEMETRY_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            logger.debug(
                "PostHog cannot be reached, so telemetry is off: %s",
                type(error).__name__,
            )
            return None
        return self._fetch_user_id()

    def _fetch_user_id(self) -> str | None:
        """Return the Hirundo user ID that owns the configured API key.

        The API key list only contains the caller's own standard keys, so their
        owner is the current user. This is the same ID the web app sends to
        PostHog, so SDK and web app events line up.
        """
        try:
            response = self._session.get(
                f"{API_HOST}/api-key/",
                headers=get_headers(),
                timeout=TELEMETRY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            api_keys = _API_KEY_OWNERS.validate_json(response.content)
        except (requests.RequestException, ValueError) as error:
            logger.debug(
                "Could not look up the user ID, so telemetry is off: %s",
                type(error).__name__,
            )
            return None
        owner_ids = {
            api_key.user_owner_id
            for api_key in api_keys
            if api_key.kind == STANDARD_API_KEY_KIND
        }
        if len(owner_ids) != 1:
            logger.debug("Could not tell the user ID, so telemetry is off")
            return None
        return str(owner_ids.pop())

    def _send(
        self,
        event: _PendingEvent,
        user_id: str,
        common_properties: CommonProperties,
    ) -> None:
        payload = {
            "api_key": self._project_key,
            "event": event.name,
            "distinct_id": user_id,
            "timestamp": event.timestamp.isoformat(),
            "properties": {
                **common_properties.model_dump(mode="json"),
                **event.properties.model_dump(mode="json"),
                "$geoip_disable": True,
            },
        }
        try:
            response = self._session.post(
                f"{self._settings.posthog_root}{POSTHOG_CAPTURE_PATH}",
                json=payload,
                timeout=TELEMETRY_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            logger.debug("Telemetry event not sent: %s", type(error).__name__)
            return
        if not response.ok:
            logger.debug(
                "PostHog rejected a telemetry event with HTTP %s, so it was dropped",
                response.status_code,
            )


def _start_reporter() -> _Reporter | None:
    settings = TelemetrySettings.from_environment()
    if settings is None or settings.opted_out or settings.posthog_project_key is None:
        return None
    return _Reporter(settings, settings.posthog_project_key)


class _ReporterState:
    """Starts the reporter at most once per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._reporter: _Reporter | None = None

    def get(self) -> _Reporter | None:
        with self._lock:
            if not self._started:
                self._started = True
                self._reporter = _start_reporter()
            return self._reporter

    def flush(self, timeout_seconds: float = TELEMETRY_TIMEOUT_SECONDS) -> None:
        with self._lock:
            reporter = self._reporter
        if reporter is not None:
            reporter.flush(timeout_seconds)


_STATE = _ReporterState()


def _flush_at_exit() -> None:
    _STATE.flush()


atexit.register(_flush_at_exit)


def _submit(name: str, build_properties: Callable[[], BaseModel]) -> None:
    try:
        reporter = _STATE.get()
        if reporter is not None:
            reporter.submit(
                _PendingEvent(
                    name=name,
                    properties=build_properties(),
                    timestamp=datetime.now(timezone.utc),
                )
            )
    except Exception as error:
        logger.debug("Telemetry event dropped: %s", type(error).__name__)


def _http_status(error: Exception) -> int | None:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code
    return None


def report_run_launched(product: Product, run_id: str) -> None:
    _submit(
        RUN_LAUNCHED_EVENT,
        lambda: RunLaunchedProperties(product=product, run_id=run_id),
    )


def report_cli_command(command: str) -> None:
    _submit(CLI_COMMAND_EVENT, lambda: CliCommandProperties(command=command))


def _report_error(product: Product, function_name: str, error: Exception) -> None:
    _submit(
        ERROR_EVENT,
        lambda: ErrorProperties(
            product=product,
            function=function_name,
            error_type=type(error).__name__,
            http_status=_http_status(error),
        ),
    )


def reports_errors(
    product: Product,
) -> Callable[[Callable[FunctionParams, ReturnT]], Callable[FunctionParams, ReturnT]]:
    """Report errors raised by a public SDK function, then re-raise them unchanged."""

    def decorator(
        function: Callable[FunctionParams, ReturnT],
    ) -> Callable[FunctionParams, ReturnT]:
        @functools.wraps(function)
        def wrapper(
            *args: FunctionParams.args, **kwargs: FunctionParams.kwargs
        ) -> ReturnT:
            try:
                return function(*args, **kwargs)
            except Exception as error:
                _report_error(product, function.__qualname__, error)
                raise

        return wrapper

    return decorator
