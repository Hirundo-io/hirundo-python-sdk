import secrets
from pathlib import Path
from typing import Any, TypeVar, cast

import hirundo._model_access as model_access_module
import httpx
import pytest
from hirundo._hirundo_error import HirundoError
from hirundo._model_access import (
    validate_huggingface_model_access,
    validate_judge_model_access,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from requests import HTTPError, Response


def _build_http_error(status_code: int) -> HTTPError:
    response_object = Response()
    response_object.status_code = status_code
    return HTTPError(response=response_object)


def _build_huggingface_error_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("GET", "https://huggingface.co/api/models/test"),
    )


ExceptionType = TypeVar("ExceptionType", bound=HfHubHTTPError)


def _build_huggingface_hub_error(
    exception_type: type[ExceptionType],
    message: str,
    status_code: int,
) -> ExceptionType:
    return cast(Any, exception_type)(
        message,
        response=_build_huggingface_error_response(status_code=status_code),
    )


def test_validate_huggingface_model_access_allows_public_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_repo_ids: list[str] = []

    def fake_model_info(_self: object, *, repo_id: str) -> None:
        captured_repo_ids.append(repo_id)

    monkeypatch.setattr(model_access_module.HfApi, "model_info", fake_model_info)

    validate_huggingface_model_access(
        model_name="some/model",
        token=None,
        model_role="LLM",
    )

    assert captured_repo_ids == ["some/model"]


def test_validate_huggingface_model_access_raises_gated_message_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_model_info(_self: object, *, repo_id: str) -> None:
        raise _build_huggingface_hub_error(GatedRepoError, "gated", status_code=403)

    monkeypatch.setattr(model_access_module.HfApi, "model_info", fake_model_info)

    with pytest.raises(HirundoError, match="is gated"):
        validate_huggingface_model_access(
            model_name="some/model",
            token=None,
            model_role="judge",
        )


def test_validate_huggingface_model_access_raises_not_found_message_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_access_secret = secrets.token_hex(8)

    def fake_model_info(_self: object, *, repo_id: str) -> None:
        raise _build_huggingface_hub_error(
            RepositoryNotFoundError, "missing", status_code=404
        )

    monkeypatch.setattr(model_access_module.HfApi, "model_info", fake_model_info)

    with pytest.raises(HirundoError, match="provided token"):
        validate_huggingface_model_access(
            model_name="missing/model",
            token=user_access_secret,
            model_role="LLM",
        )


def test_validate_huggingface_model_access_raises_unauthorized_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_access_secret = secrets.token_hex(8)

    def fake_model_info(_self: object, *, repo_id: str) -> None:
        raise _build_http_error(status_code=401)

    monkeypatch.setattr(model_access_module.HfApi, "model_info", fake_model_info)

    with pytest.raises(HirundoError, match="provided HuggingFace token"):
        validate_huggingface_model_access(
            model_name="private/model",
            token=user_access_secret,
            model_role="judge",
        )


def test_validate_judge_model_access_skips_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_model_path = tmp_path / "local_model"
    local_model_path.mkdir()
    validate_calls: list[tuple[str, str | None, str]] = []

    def fake_validate_huggingface_model_access(
        model_name: str, token: str | None, model_role: str
    ) -> None:
        validate_calls.append((model_name, token, model_role))

    monkeypatch.setattr(
        model_access_module,
        "validate_huggingface_model_access",
        fake_validate_huggingface_model_access,
    )

    validate_judge_model_access(str(local_model_path), token=None)

    assert validate_calls == []
