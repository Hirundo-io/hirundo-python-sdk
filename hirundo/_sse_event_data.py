from pydantic import BaseModel, ValidationError

from hirundo import HirundoError
from hirundo._run_status import RunStatus
from hirundo.logger import get_logger

logger = get_logger(__name__)


class SseRunEventData(BaseModel):
    id: str
    state: RunStatus
    result: str | dict | None


class SseRunEventDataPayload(BaseModel):
    data: SseRunEventData


def _parse_sse_payload(payload: str) -> SseRunEventData:
    try:
        return SseRunEventDataPayload.model_validate_json(payload).data
    except ValidationError as e:
        logger.error("Invalid SSE payload: %s: %s", payload, exc_info=e)
        raise HirundoError(f"Invalid SSE payload: {payload}") from e
