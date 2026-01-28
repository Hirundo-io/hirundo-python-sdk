from typing import Any

from pydantic import BaseModel

from hirundo._run_status import RunStatus
from hirundo.logger import get_logger

logger = get_logger(__name__)


class SseRunEventData(BaseModel):
    id: str
    state: RunStatus
    result: str | dict | None


def _parse_sse_payload(payload: Any) -> SseRunEventData:
    if isinstance(payload, dict):
        if "data" in payload:
            data = payload["data"]
            if isinstance(data, dict):
                return SseRunEventData.model_validate(data)

    logger.error("Invalid SSE payload: %s", payload)
    raise ValueError(f"Invalid SSE payload: {payload}")
