import json
from collections.abc import AsyncGenerator, Generator
from enum import Enum

import httpx
from tqdm import tqdm

from hirundo._iter_sse_retrying import aiter_sse_retrying, iter_sse_retrying
from hirundo.logger import get_logger

_logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 200


class RunStatus(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    AWAITING_MANUAL_APPROVAL = "AWAITING MANUAL APPROVAL"
    REVOKED = "REVOKED"
    REJECTED = "REJECTED"
    RETRY = "RETRY"


STATUS_TO_PROGRESS_MAP = {
    RunStatus.STARTED.value: 0.0,
    RunStatus.PENDING.value: 0.0,
    RunStatus.SUCCESS.value: 100.0,
    RunStatus.FAILURE.value: 100.0,
    RunStatus.AWAITING_MANUAL_APPROVAL.value: 100.0,
    RunStatus.RETRY.value: 0.0,
    RunStatus.REVOKED.value: 100.0,
    RunStatus.REJECTED.value: 0.0,
}


def build_status_text_map(
    run_label: str, *, started_detail: str | None = None
) -> dict[str, str]:
    """
    Build a status->text mapping for a given run label.

    Args:
        run_label: Human-readable label used in status text.
        started_detail: Optional override for the STARTED status text.

    Returns:
        Mapping of run state values to user-facing status text.
    """
    started_text = started_detail or f"{run_label} run in progress"
    return {
        RunStatus.STARTED.value: started_text,
        RunStatus.PENDING.value: f"{run_label} run queued and not yet started",
        RunStatus.SUCCESS.value: f"{run_label} run completed successfully",
        RunStatus.FAILURE.value: f"{run_label} run failed",
        RunStatus.AWAITING_MANUAL_APPROVAL.value: "Awaiting manual approval",
        RunStatus.RETRY.value: f"{run_label} run failed. Retrying",
        RunStatus.REVOKED.value: f"{run_label} run was cancelled",
        RunStatus.REJECTED.value: f"{run_label} run was rejected",
    }


def get_state(payload: dict, status_keys: tuple[str, ...]) -> str | None:
    """
    Return the first non-null state value from a payload using a list of keys.

    Args:
        payload: Run payload containing state/status information.
        status_keys: Ordered keys to search for state values.

    Returns:
        The first non-null state value, or None if none are present.
    """
    for key in status_keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def iter_run_events(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retry: int = 0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    pending_state_value: str = RunStatus.PENDING.value,
    status_keys: tuple[str, ...] = ("state",),
    error_cls: type[Exception] = RuntimeError,
    log=_logger,
) -> Generator[dict, None, None]:
    """
    Stream run events from an SSE endpoint with retries.

    Args:
        url: SSE endpoint URL.
        headers: Optional HTTP headers.
        retry: Internal retry counter (do not set manually).
        max_retries: Maximum number of retry attempts.
        pending_state_value: State value that triggers a re-check loop.
        status_keys: Payload keys to search for the run state.
        error_cls: Exception type to raise on errors.
        log: Logger instance for debug output.

    Yields:
        Event payloads decoded from the SSE data field.
    """
    if retry > max_retries:
        raise error_cls("Max retries reached")
    last_event = None
    with httpx.Client(timeout=httpx.Timeout(None, connect=5.0)) as client:
        for sse in iter_sse_retrying(
            client,
            "GET",
            url,
            headers=headers,
        ):
            if sse.event == "ping":
                continue
            log.debug(
                "[SYNC] received event: %s with data: %s and ID: %s and retry: %s",
                sse.event,
                sse.data,
                sse.id,
                sse.retry,
            )
            last_event = json.loads(sse.data)
            if not last_event:
                continue
            if "data" in last_event:
                data = last_event["data"]
            else:
                if "detail" in last_event:
                    raise error_cls(last_event["detail"])
                if "reason" in last_event:
                    raise error_cls(last_event["reason"])
                raise error_cls("Unknown error")
            yield data
    last_state = None
    if last_event and "data" in last_event:
        last_state = get_state(last_event["data"], status_keys)
    if not last_event or last_state == pending_state_value:
        iter_run_events(
            url,
            headers=headers,
            retry=retry + 1,
            max_retries=max_retries,
            pending_state_value=pending_state_value,
            status_keys=status_keys,
            error_cls=error_cls,
            log=log,
        )


async def aiter_run_events(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retry: int = 0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    pending_state_value: str = RunStatus.PENDING.value,
    status_keys: tuple[str, ...] = ("state",),
    error_cls: type[Exception] = RuntimeError,
    log=_logger,
) -> AsyncGenerator[dict, None]:
    """
    Async stream run events from an SSE endpoint with retries.

    Args:
        url: SSE endpoint URL.
        headers: Optional HTTP headers.
        retry: Internal retry counter (do not set manually).
        max_retries: Maximum number of retry attempts.
        pending_state_value: State value that triggers a re-check loop.
        status_keys: Payload keys to search for the run state.
        error_cls: Exception type to raise on errors.
        log: Logger instance for debug output.

    Yields:
        Event payloads decoded from the SSE data field.
    """
    if retry > max_retries:
        raise error_cls("Max retries reached")
    last_event = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as client:
        async_iterator = await aiter_sse_retrying(
            client,
            "GET",
            url,
            headers=headers or {},
        )
        async for sse in async_iterator:
            if sse.event == "ping":
                continue
            log.debug(
                "[ASYNC] Received event: %s with data: %s and ID: %s and retry: %s",
                sse.event,
                sse.data,
                sse.id,
                sse.retry,
            )
            last_event = json.loads(sse.data)
            if "data" not in last_event:
                if "detail" in last_event:
                    raise error_cls(last_event["detail"])
                if "reason" in last_event:
                    raise error_cls(last_event["reason"])
                raise error_cls("Unknown error")
            yield last_event["data"]
    last_state = None
    if last_event and "data" in last_event:
        last_state = get_state(last_event["data"], status_keys)
    if not last_event or last_state == pending_state_value:
        aiter_run_events(
            url,
            headers=headers,
            retry=retry + 1,
            max_retries=max_retries,
            pending_state_value=pending_state_value,
            status_keys=status_keys,
            error_cls=error_cls,
            log=log,
        )


def update_progress_from_result(
    iteration: dict,
    progress: tqdm,
    *,
    uploading_text: str,
    log=_logger,
) -> bool:
    """
    Update a tqdm progress bar based on a serialized progress result string.

    Args:
        iteration: Payload containing a nested result string.
        progress: tqdm instance to update.
        uploading_text: Description to show when progress reaches 100%.
        log: Logger instance for debug output.

    Returns:
        True if a progress update occurred, False otherwise.
    """
    if (
        iteration.get("result")
        and isinstance(iteration["result"], dict)
        and iteration["result"].get("result")
        and isinstance(iteration["result"]["result"], str)
    ):
        result_info = iteration["result"]["result"].split(":")
        if len(result_info) > 1:
            stage = result_info[0]
            current_progress_percentage = float(
                result_info[1].removeprefix(" ").removesuffix("% done")
            )
        elif len(result_info) == 1:
            stage = result_info[0]
            current_progress_percentage = progress.n
        else:
            stage = "Unknown progress state"
            current_progress_percentage = progress.n
        desc = uploading_text if current_progress_percentage == 100.0 else stage
        progress.set_description(desc)
        progress.n = current_progress_percentage
        log.debug("Setting progress to %s", progress.n)
        progress.refresh()
        return True
    return False


def handle_run_failure(
    iteration: dict, *, error_cls: type[Exception], run_label: str
) -> None:
    """
    Raise a run-specific failure exception based on the iteration payload.

    Args:
        iteration: Payload containing error details.
        error_cls: Exception type to raise.
        run_label: Human-readable label for the run type.
    """
    if iteration.get("result"):
        raise error_cls(f"{run_label} run failed with error: {iteration['result']}")
    raise error_cls(f"{run_label} run failed with an unknown error")
