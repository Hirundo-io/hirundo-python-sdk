"""Validate API responses with the models generated from the OpenAPI schema."""

from typing import Annotated, TypeVar

from pydantic import BaseModel, StringConstraints
from requests import Response

from hirundo._generated.wire_models import CreateModel, LaunchRunResponse
from hirundo._hirundo_error import HirundoError

NonEmptyRunId = Annotated[str, StringConstraints(min_length=1)]

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LaunchedRunResponse(LaunchRunResponse):
    """Generated launch response that also rejects an empty run ID."""

    run_id: NonEmptyRunId


def parse_api_response(
    response: Response,
    response_model: type[ResponseModel],
    error_cls: type[HirundoError] = HirundoError,
) -> ResponseModel:
    """Validate a successful API response body against a response model.

    Args:
        response: Successful HTTP response returned by the Hirundo API.
        response_model: Pydantic model that describes the response body.
        error_cls: Exception type raised when the body does not match the model.

    Returns:
        The validated response model.

    Raises:
        HirundoError: If the body is not JSON or does not match `response_model`.
            The raised type is `error_cls`.
    """
    try:
        return response_model.model_validate(response.json())
    except ValueError as error:
        raise error_cls(
            f"The API returned an invalid {response_model.__name__} response: {error}"
        ) from error


def parse_created_id(
    response: Response, error_cls: type[HirundoError] = HirundoError
) -> int:
    """Return the ID from a create response validated as `CreateModel`.

    Args:
        response: Successful HTTP response returned by a create endpoint.
        error_cls: Exception type raised when the body is not a `CreateModel`.

    Returns:
        The ID of the created resource.
    """
    return parse_api_response(response, CreateModel, error_cls).id


def parse_launched_run_id(
    response: Response, error_cls: type[HirundoError] = HirundoError
) -> str:
    """Return the non-empty run ID from a validated launch response.

    Args:
        response: Successful HTTP response returned by a run launch endpoint.
        error_cls: Exception type raised when the body has no usable run ID.

    Returns:
        The ID of the launched run.
    """
    return parse_api_response(response, LaunchedRunResponse, error_cls).run_id
