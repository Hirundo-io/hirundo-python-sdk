from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
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


def validate_huggingface_model_access(
    model_name: str,
    token: str | None,
    model_role: str,
) -> None:
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
    except HTTPError as exception:
        if exception.response is not None and exception.response.status_code in {
            401,
            403,
        }:
            hint = "unauthorized"
        else:
            hint = "generic"
        logger.debug(
            "HuggingFace access validation failed for %s model '%s' with status %s.",
            model_role,
            model_name,
            exception.response.status_code if exception.response is not None else None,
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
    if _is_local_model_path(path_or_repo_id):
        return

    validate_huggingface_model_access(
        model_name=path_or_repo_id,
        token=token,
        model_role="judge",
    )
