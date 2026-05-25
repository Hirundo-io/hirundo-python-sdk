from typing import TYPE_CHECKING, cast

import hirundo._iter_sse_retrying as sse_retrying
import httpx
import pytest
from httpx_sse import ServerSentEvent
from urllib3.exceptions import ReadTimeoutError

if TYPE_CHECKING:
    from urllib3.connectionpool import ConnectionPool


SSE_URL = "https://example.test/events"


class SyncEventSource:
    def __init__(
        self,
        *,
        event: ServerSentEvent | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.event = event
        self.exception = exception

    def __enter__(self) -> "SyncEventSource":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def iter_sse(self):
        if self.exception is not None:
            raise self.exception
        if self.event is not None:
            yield self.event


class AsyncEventSource:
    def __init__(
        self,
        *,
        event: ServerSentEvent | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.event = event
        self.exception = exception

    async def __aenter__(self) -> "AsyncEventSource":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def aiter_sse(self):
        if self.exception is not None:
            raise self.exception
        if self.event is not None:
            yield self.event


def _read_timeout_error() -> ReadTimeoutError:
    return ReadTimeoutError(
        cast("ConnectionPool", None),
        SSE_URL,
        "timed out",
    )


def _disable_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sse_retrying, "MAX_RETRIES", 2)
    monkeypatch.setattr(sse_retrying.time, "sleep", lambda _seconds: None)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sse_retrying.asyncio, "sleep", fake_sleep)


def test_iter_sse_retrying_retries_read_timeout(monkeypatch: pytest.MonkeyPatch):
    _disable_retry_delays(monkeypatch)
    expected_event = ServerSentEvent(data="done", event="message", id="event-1")
    event_sources = [
        SyncEventSource(exception=_read_timeout_error()),
        SyncEventSource(event=expected_event),
    ]
    connect_call_count = 0

    def fake_connect_sse(
        _client: httpx.Client,
        _method_name: str,
        _url: str,
        headers: dict[str, str],
    ) -> SyncEventSource:
        _ = headers
        nonlocal connect_call_count
        connect_call_count += 1
        return event_sources.pop(0)

    monkeypatch.setattr(sse_retrying, "connect_sse", fake_connect_sse)

    with httpx.Client() as client:
        events = list(
            sse_retrying.iter_sse_retrying(
                client,
                "GET",
                SSE_URL,
            )
        )

    assert events == [expected_event]
    assert connect_call_count == 2


def test_iter_sse_retrying_raises_after_read_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    _disable_retry_delays(monkeypatch)
    connect_call_count = 0

    def fake_connect_sse(
        _client: httpx.Client,
        _method_name: str,
        _url: str,
        headers: dict[str, str],
    ) -> SyncEventSource:
        _ = headers
        nonlocal connect_call_count
        connect_call_count += 1
        return SyncEventSource(exception=_read_timeout_error())

    monkeypatch.setattr(sse_retrying, "connect_sse", fake_connect_sse)

    with httpx.Client() as client:
        with pytest.raises(ReadTimeoutError):
            list(
                sse_retrying.iter_sse_retrying(
                    client,
                    "GET",
                    SSE_URL,
                )
            )

    assert connect_call_count == 2


@pytest.mark.asyncio
async def test_aiter_sse_retrying_retries_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    _disable_retry_delays(monkeypatch)
    expected_event = ServerSentEvent(data="done", event="message", id="event-1")
    event_sources = [
        AsyncEventSource(exception=_read_timeout_error()),
        AsyncEventSource(event=expected_event),
    ]
    connect_call_count = 0

    def fake_aconnect_sse(
        _client: httpx.AsyncClient,
        _method_name: str,
        _url: str,
        headers: dict[str, str],
    ) -> AsyncEventSource:
        _ = headers
        nonlocal connect_call_count
        connect_call_count += 1
        return event_sources.pop(0)

    monkeypatch.setattr(sse_retrying, "aconnect_sse", fake_aconnect_sse)

    async with httpx.AsyncClient() as client:
        event_iterator = await sse_retrying.aiter_sse_retrying(
            client,
            "GET",
            SSE_URL,
            headers={},
        )
        events = [event async for event in event_iterator]

    assert events == [expected_event]
    assert connect_call_count == 2


@pytest.mark.asyncio
async def test_aiter_sse_retrying_raises_after_read_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    _disable_retry_delays(monkeypatch)
    connect_call_count = 0

    def fake_aconnect_sse(
        _client: httpx.AsyncClient,
        _method_name: str,
        _url: str,
        headers: dict[str, str],
    ) -> AsyncEventSource:
        _ = headers
        nonlocal connect_call_count
        connect_call_count += 1
        return AsyncEventSource(exception=_read_timeout_error())

    monkeypatch.setattr(sse_retrying, "aconnect_sse", fake_aconnect_sse)

    async with httpx.AsyncClient() as client:
        event_iterator = await sse_retrying.aiter_sse_retrying(
            client,
            "GET",
            SSE_URL,
            headers={},
        )
        with pytest.raises(ReadTimeoutError):
            [event async for event in event_iterator]

    assert connect_call_count == 2
