from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from requests import HTTPError

from hirundo._hirundo_error import HirundoError
from hirundo.logger import get_logger

logger = get_logger(__name__)


def _build_huggingface_access_message(
    model_name: str,
    model_role: str,
    hint: str,
    token_provided: bool,
) -> str:
    message_prefix = f"The {model_role} model '{model_name}'"

    if hint == "gated":
        if token_provided:
            return (
                f"{message_prefix} is gated and the provided HuggingFace token does not "
                "have access. Please request access or use a different token."
            )
        return f"{message_prefix} is gated. Please provide a HuggingFace token with access."

    if hint == "not_found":
        if token_provided:
            return (
                f"{message_prefix} was not found or is private/gated for the provided "
                "token. Please verify the model ID or token access."
            )
        return (
            f"{message_prefix} was not found or is private/gated. Please provide a "
            "HuggingFace token or verify the model ID."
        )

    if hint == "unauthorized":
        if token_provided:
            return (
                f"{message_prefix} could not be accessed with the provided HuggingFace "
                "token. Please verify token permissions or use a different model."
            )
        return (
            f"{message_prefix} could not be accessed without a HuggingFace token. "
            "Please provide a token or use a public model."
        )

    return (
        f"{message_prefix} could not be accessed. Please verify the model ID or provide "
        "a HuggingFace token with access."
    )


def _is_local_model_path(path_or_repo_id: str) -> bool:
    potential_path = Path(path_or_repo_id).expanduser()
    return potential_path.exists()


def _get_huggingface_error_status_code(
    exception: HfHubHTTPError | HTTPError,
) -> int | None:
    response = exception.response
    return response.status_code if response is not None else None


def validate_huggingface_model_access(
    model_name: str,
    token: str | None,
    model_role: str,
) -> None:
    """Validate that a Hugging Face model can be accessed.

    Args:
        model_name: Hugging Face repository ID for the model to validate.
        token: Optional Hugging Face access token used for authenticated access.
        model_role: Human-readable role for the model in error messages.
    """
    huggingface_api = HfApi(token=token)
    token_provided = token is not None

    try:
        huggingface_api.model_info(repo_id=model_name)
    except GatedRepoError as exception:
        raise HirundoError(
            _build_huggingface_access_message(
                model_name=model_name,
                model_role=model_role,
                hint="gated",
                token_provided=token_provided,
            )
        ) from exception
    except RepositoryNotFoundError as exception:
        raise HirundoError(
            _build_huggingface_access_message(
                model_name=model_name,
                model_role=model_role,
                hint="not_found",
                token_provided=token_provided,
            )
        ) from exception
    except (HfHubHTTPError, HTTPError) as exception:
        status_code = _get_huggingface_error_status_code(exception)

        if status_code in {401, 403}:
            hint = "unauthorized"
        else:
            hint = "generic"
        logger.debug(
            "HuggingFace access validation failed for %s model '%s' with status %s.",
            model_role,
            model_name,
            status_code,
        )
        raise HirundoError(
            _build_huggingface_access_message(
                model_name=model_name,
                model_role=model_role,
                hint=hint,
                token_provided=token_provided,
            )
        ) from exception


def validate_judge_model_access(path_or_repo_id: str, token: str | None) -> None:
    """Validate that a judge model can be accessed.

    Args:
        path_or_repo_id: Local filesystem path or Hugging Face repository ID for the
            judge model.
        token: Optional Hugging Face access token used when the judge model is hosted
            on Hugging Face.
    """
    if _is_local_model_path(path_or_repo_id):
        return

    validate_huggingface_model_access(
        model_name=path_or_repo_id,
        token=token,
        model_role="judge",
    )
