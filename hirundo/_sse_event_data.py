from typing import Any

from pydantic import BaseModel, TypeAdapter

from hirundo._run_checking import RunStatus


class SseRunEventData(BaseModel):
    id: str
    state: RunStatus
    result: str | dict | None


_SSE_RUN_EVENT_ADAPTER = TypeAdapter(SseRunEventData)


def _parse_sse_payload(payload: Any) -> SseRunEventData:
    if isinstance(payload, dict):
        if "data" in payload:
            data = payload["data"]
            if isinstance(data, dict):
                return _SSE_RUN_EVENT_ADAPTER.validate_python(data)

    raise ValueError(f"Invalid SSE payload: {payload}")
