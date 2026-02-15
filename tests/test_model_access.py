import secrets
from pathlib import Path
from unittest.mock import patch

import pytest
from hirundo._hirundo_error import HirundoError
from hirundo._model_access import (
    validate_huggingface_model_access,
    validate_judge_model_access,
)
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from requests import HTTPError, Response


def _build_http_error(status_code: int) -> HTTPError:
    response_object = Response()
    response_object.status_code = status_code
    return HTTPError(response=response_object)


def test_validate_huggingface_model_access_allows_public_model() -> None:
    with patch("hirundo._model_access.HfApi.model_info") as mock_model_info:
        validate_huggingface_model_access(
            model_name="some/model",
            token=None,
            model_role="LLM",
        )

    mock_model_info.assert_called_once_with(repo_id="some/model")


def test_validate_huggingface_model_access_raises_gated_message_without_token() -> None:
    with patch(
        "hirundo._model_access.HfApi.model_info",
        side_effect=GatedRepoError("gated"),
    ):
        with pytest.raises(HirundoError, match="is gated"):
            validate_huggingface_model_access(
                model_name="some/model",
                token=None,
                model_role="judge",
            )


def test_validate_huggingface_model_access_raises_not_found_message_with_token() -> (
    None
):
    user_access_secret = secrets.token_hex(8)
    with patch(
        "hirundo._model_access.HfApi.model_info",
        side_effect=RepositoryNotFoundError("missing"),
    ):
        with pytest.raises(HirundoError, match="provided token"):
            validate_huggingface_model_access(
                model_name="missing/model",
                token=user_access_secret,
                model_role="LLM",
            )


def test_validate_huggingface_model_access_raises_unauthorized_message() -> None:
    user_access_secret = secrets.token_hex(8)
    with patch(
        "hirundo._model_access.HfApi.model_info",
        side_effect=_build_http_error(status_code=401),
    ):
        with pytest.raises(HirundoError, match="provided HuggingFace token"):
            validate_huggingface_model_access(
                model_name="private/model",
                token=user_access_secret,
                model_role="judge",
            )


def test_validate_judge_model_access_skips_local_path(tmp_path: Path) -> None:
    local_model_path = tmp_path / "local_model"
    local_model_path.mkdir()

    with patch(
        "hirundo._model_access.validate_huggingface_model_access"
    ) as mock_validate:
        validate_judge_model_access(str(local_model_path), token=None)

    mock_validate.assert_not_called()
