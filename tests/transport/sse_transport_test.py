import httpx
import pytest
from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from tests.transport.server import LocalTransportServer


def test_sync_sse_delivers_an_event_before_reconnecting_after_disconnect(
    local_transport_server: LocalTransportServer,
) -> None:
    with httpx.Client(timeout=1.0) as client:
        event_iterator = iter_sse_retrying(
            client,
            "GET",
            f"{local_transport_server.base_url}/events",
        )

        first_event = next(event_iterator)
        assert first_event.id == "event-1"
        assert first_event.event == "progress"
        assert first_event.data == "first\nline"
        assert local_transport_server.request_counts["/events"] == 1

        local_transport_server.disconnect_sse.set()
        remaining_events = list(event_iterator)

    assert [(event.id, event.event, event.data) for event in remaining_events] == [
        ("event-2", "complete", "done")
    ]
    assert local_transport_server.request_counts["/events"] == 2
    second_headers = local_transport_server.request_headers["/events"][1]
    assert second_headers["Accept"] == "text/event-stream"
    assert second_headers["Last-Event-ID"] == "event-1"


@pytest.mark.asyncio
async def test_async_sse_delivers_an_event_before_reconnecting_after_disconnect(
    local_transport_server: LocalTransportServer,
) -> None:
    async with httpx.AsyncClient(timeout=1.0) as client:
        event_iterator = await aiter_sse_retrying(
            client,
            "GET",
            f"{local_transport_server.base_url}/events",
            headers={},
        )

        first_event = await anext(event_iterator)
        assert first_event.id == "event-1"
        assert first_event.event == "progress"
        assert first_event.data == "first\nline"
        assert local_transport_server.request_counts["/events"] == 1

        local_transport_server.disconnect_sse.set()
        remaining_events = [event async for event in event_iterator]

    assert [(event.id, event.event, event.data) for event in remaining_events] == [
        ("event-2", "complete", "done")
    ]
    assert local_transport_server.request_counts["/events"] == 2
    second_headers = local_transport_server.request_headers["/events"][1]
    assert second_headers["Accept"] == "text/event-stream"
    assert second_headers["Last-Event-ID"] == "event-1"
