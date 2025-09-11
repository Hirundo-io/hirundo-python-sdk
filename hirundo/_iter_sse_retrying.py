import asyncio
import time
import typing
import uuid
from collections.abc import AsyncGenerator, Generator

import httpx
import urllib3
from httpx_sse import ServerSentEvent, SSEError, aconnect_sse, connect_sse
from stamina import retry

from hirundo._http import requests
from hirundo._timeouts import READ_TIMEOUT
from hirundo.logger import get_logger

logger = get_logger(__name__)


# Credit: https://github.com/florimondmanca/httpx-sse/blob/master/README.md#handling-reconnections
def iter_sse_retrying(
    client: httpx.Client,
    method: str,
    url: str,
    headers: typing.Optional[dict[str, str]] = None,
) -> Generator[ServerSentEvent, None, None]:
    if headers is None:
        headers = {}
    last_event_id = ""
    reconnection_delay = 0.0

    # `stamina` will apply jitter and exponential backoff on top of
    # the `retry` reconnection delay sent by the server.
    # httpx.ReadError is thrown when there is a network error.
    #   Some network errors may be temporary, hence the retries.
    # httpx.RemoteProtocolError is thrown when the server closes the connection.
    #  This may happen when the server is overloaded and closes the connection or
    #  when Kubernetes restarts / replaces a pod.
    #  Likewise, this will likely be temporary, hence the retries.
    @retry(
        on=(
            httpx.ReadError,
            httpx.RemoteProtocolError,
            urllib3.exceptions.ReadTimeoutError,
        )
    )
    def _iter_sse():
        nonlocal last_event_id, reconnection_delay

        time.sleep(reconnection_delay)

        connect_headers = {
            **headers,
            "Accept": "text/event-stream",
            "X-Accel-Buffering": "no",
        }

        if last_event_id:
            connect_headers["Last-Event-ID"] = last_event_id

        with connect_sse(client, method, url, headers=connect_headers) as event_source:
            try:
                for sse in event_source.iter_sse():
                    last_event_id = sse.id

                    if sse.retry is not None:
                        reconnection_delay = sse.retry / 1000

                    yield sse
            except SSEError:
                logger.error("SSE error occurred. Trying regular request")
                response = requests.get(
                    url,
                    headers=connect_headers,
                    timeout=READ_TIMEOUT,
                )
                yield ServerSentEvent(
                    event="",
                    data=response.text,
                    id=uuid.uuid4().hex,
                    retry=None,
                )

    return _iter_sse()


async def aiter_sse_retrying(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
) -> AsyncGenerator[ServerSentEvent, None]:
    last_event_id = ""
    reconnection_delay = 0.0

    # `stamina` will apply jitter and exponential backoff on top of
    # the `retry` reconnection delay sent by the server.
    # httpx.ReadError is thrown when there is a network error.
    #   Some network errors may be temporary, hence the retries.
    # httpx.RemoteProtocolError is thrown when the server closes the connection.
    #  This may happen when the server is overloaded and closes the connection or
    #  when Kubernetes restarts / replaces a pod.
    #  Likewise, this will likely be temporary, hence the retries.
    @retry(
        on=(
            httpx.ReadError,
            httpx.RemoteProtocolError,
            urllib3.exceptions.ReadTimeoutError,
        )
    )
    async def _iter_sse() -> AsyncGenerator[ServerSentEvent, None]:
        nonlocal last_event_id, reconnection_delay

        await asyncio.sleep(reconnection_delay)

        connect_headers = {**headers, "Accept": "text/event-stream"}

        if last_event_id:
            connect_headers["Last-Event-ID"] = last_event_id

        async with aconnect_sse(
            client, method, url, headers=connect_headers
        ) as event_source:
            try:
                async for sse in event_source.aiter_sse():
                    last_event_id = sse.id

                    if sse.retry is not None:
                        reconnection_delay = sse.retry / 1000

                    yield sse
            except SSEError:
                logger.error("SSE error occurred. Trying regular request")
                response = await client.get(url, headers=connect_headers)
                yield ServerSentEvent(
                    event="",
                    data=response.text,
                    id=uuid.uuid4().hex,
                    retry=None,
                )

    return _iter_sse()
