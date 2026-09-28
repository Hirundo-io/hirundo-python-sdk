from pydantic import BaseModel, ValidationError

from hirundo._hirundo_error import HirundoError
from hirundo._run_status import RunStatus
from hirundo.logger import get_logger

logger = get_logger(__name__)


class SseRunEventData(BaseModel):
    id: str
    state: RunStatus | None
    result: str | dict | None


class SseRunEventDataPayload(BaseModel):
    data: SseRunEventData


def _parse_sse_payload(payload: str) -> SseRunEventData:
    try:
        return SseRunEventDataPayload.model_validate_json(payload).data
    except ValidationError as validation_error:
        logger.warning("Invalid SSE payload received from the API.")
        raise HirundoError("Invalid SSE payload received from the API.") from validation_error
