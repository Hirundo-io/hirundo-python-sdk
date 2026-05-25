from typing import TYPE_CHECKING, cast

import hirundo._iter_sse_retrying as sse_retrying
import httpx
import pytest
from httpx_sse import ServerSentEvent
from urllib3.exceptions import ReadTimeoutError

if TYPE_CHECKING:
    from urllib3.connectionpool import ConnectionPool


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
        "https://example.test/events",
        "timed out",
    )


def test_iter_sse_retrying_retries_read_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sse_retrying, "MAX_RETRIES", 2)
    monkeypatch.setattr(sse_retrying.time, "sleep", lambda _seconds: None)

    expected_event = ServerSentEvent(data="done", event="message", id="event-1")
    event_sources = [
        SyncEventSource(exception=_read_timeout_error()),
        SyncEventSource(event=expected_event),
    ]
    connect_calls = []

    def fake_connect_sse(
        client: httpx.Client,
        method_name: str,
        url: str,
        headers: dict[str, str],
    ) -> SyncEventSource:
        connect_calls.append((client, method_name, url, headers))
        return event_sources.pop(0)

    monkeypatch.setattr(sse_retrying, "connect_sse", fake_connect_sse)

    with httpx.Client() as client:
        events = list(
            sse_retrying.iter_sse_retrying(
                client,
                "GET",
                "https://example.test/events",
            )
        )

    assert events == [expected_event]
    assert len(connect_calls) == 2


def test_iter_sse_retrying_raises_after_read_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sse_retrying, "MAX_RETRIES", 2)
    monkeypatch.setattr(sse_retrying.time, "sleep", lambda _seconds: None)
    connect_calls = []

    def fake_connect_sse(
        client: httpx.Client,
        method_name: str,
        url: str,
        headers: dict[str, str],
    ) -> SyncEventSource:
        connect_calls.append((client, method_name, url, headers))
        return SyncEventSource(exception=_read_timeout_error())

    monkeypatch.setattr(sse_retrying, "connect_sse", fake_connect_sse)

    with httpx.Client() as client:
        with pytest.raises(ReadTimeoutError):
            list(
                sse_retrying.iter_sse_retrying(
                    client,
                    "GET",
                    "https://example.test/events",
                )
            )

    assert len(connect_calls) == 2


@pytest.mark.asyncio
async def test_aiter_sse_retrying_retries_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sse_retrying, "MAX_RETRIES", 2)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sse_retrying.asyncio, "sleep", fake_sleep)

    expected_event = ServerSentEvent(data="done", event="message", id="event-1")
    event_sources = [
        AsyncEventSource(exception=_read_timeout_error()),
        AsyncEventSource(event=expected_event),
    ]
    connect_calls = []

    def fake_aconnect_sse(
        client: httpx.AsyncClient,
        method_name: str,
        url: str,
        headers: dict[str, str],
    ) -> AsyncEventSource:
        connect_calls.append((client, method_name, url, headers))
        return event_sources.pop(0)

    monkeypatch.setattr(sse_retrying, "aconnect_sse", fake_aconnect_sse)

    async with httpx.AsyncClient() as client:
        event_iterator = await sse_retrying.aiter_sse_retrying(
            client,
            "GET",
            "https://example.test/events",
            headers={},
        )
        events = [event async for event in event_iterator]

    assert events == [expected_event]
    assert len(connect_calls) == 2


@pytest.mark.asyncio
async def test_aiter_sse_retrying_raises_after_read_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sse_retrying, "MAX_RETRIES", 2)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sse_retrying.asyncio, "sleep", fake_sleep)
    connect_calls = []

    def fake_aconnect_sse(
        client: httpx.AsyncClient,
        method_name: str,
        url: str,
        headers: dict[str, str],
    ) -> AsyncEventSource:
        connect_calls.append((client, method_name, url, headers))
        return AsyncEventSource(exception=_read_timeout_error())

    monkeypatch.setattr(sse_retrying, "aconnect_sse", fake_aconnect_sse)

    async with httpx.AsyncClient() as client:
        event_iterator = await sse_retrying.aiter_sse_retrying(
            client,
            "GET",
            "https://example.test/events",
            headers={},
        )
        with pytest.raises(ReadTimeoutError):
            [event async for event in event_iterator]

    assert len(connect_calls) == 2
