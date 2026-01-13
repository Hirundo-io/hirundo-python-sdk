"""Utilities for interacting with Hirundo large language model APIs."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT


class BiasType(StrEnum):
    """Types of bias mitigations supported by the Hirundo API."""

    ALL = "ALL"
    HATE_SPEECH = "HATE_SPEECH"
    SEXUAL_CONTENT = "SEXUAL_CONTENT"
    HARASSMENT = "HARASSMENT"
    SELF_HARM = "SELF_HARM"
    VIOLENCE = "VIOLENCE"
    OTHER = "OTHER"


class BiasRunInfo(BaseModel):
    """Configuration for an unlearning bias mitigation run."""

    bias_type: BiasType | str
    unlearning_data_ids: list[int] | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")

    def to_payload(self) -> dict[str, Any]:
        """Serialize the run information for use in HTTP requests."""

        return self.model_dump(mode="json", exclude_none=True)


class HuggingFaceTransformersModel(BaseModel):
    """A Hugging Face transformers model source."""

    source_type: str = Field(
        default="HUGGING_FACE_TRANSFORMERS",
        validation_alias="model_source_type",
        serialization_alias="model_source_type",
    )
    model_name: str
    token: str

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class LlmModel(BaseModel):
    """Representation of an LLM managed through Hirundo."""

    id: int | None = None
    model_name: str
    model_source: HuggingFaceTransformersModel
    description: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def create(
        self,
        organization_id: int | None = None,
        replace_if_exists: bool = False,
    ) -> int:
        """Create the LLM model on the Hirundo server."""

        payload = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        payload.pop("id", None)
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if replace_if_exists:
            payload["replace_if_exists"] = replace_if_exists

        response = requests.post(
            f"{API_HOST}/llm/model/",
            json=payload,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_json = response.json()
        self.id = response_json.get("id")
        if self.id is None:
            raise ValueError("Server response did not include a model ID")
        return self.id

    @staticmethod
    def get_by_id(model_id: int) -> LlmModel:
        """Fetch an existing LLM model from the Hirundo server."""

        response = requests.get(
            f"{API_HOST}/llm/model/{model_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        return LlmModel(**response.json())

    @staticmethod
    def list_models(organization_id: int | None = None) -> list[LlmModel]:
        """List LLM models available to the authenticated organization."""

        response = requests.get(
            f"{API_HOST}/llm/model/",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
            params={"organization_id": organization_id} if organization_id else None,
        )
        raise_for_status_with_reason(response)
        return [LlmModel(**model_json) for model_json in response.json()]

    @staticmethod
    def delete_by_id(model_id: int) -> None:
        """Delete an existing LLM model from the Hirundo server."""

        response = requests.delete(
            f"{API_HOST}/llm/model/{model_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def delete(self) -> None:
        """Delete the model represented by this instance."""

        if self.id is None:
            raise ValueError("Model must have an ID before it can be deleted")
        self.delete_by_id(self.id)


class UnlearningExample(BaseModel):
    """An individual example used for LLM unlearning."""

    prompt: str
    completion: str | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class LlmUnlearningData(BaseModel):
    """A dataset containing examples for LLM unlearning."""

    id: int | None = None
    name: str
    description: str | None = None
    examples: list[UnlearningExample]
    tags: list[str] | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def create(
        self,
        organization_id: int | None = None,
        replace_if_exists: bool = False,
    ) -> int:
        """Create the unlearning dataset on the Hirundo server."""

        payload = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        payload.pop("id", None)
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if replace_if_exists:
            payload["replace_if_exists"] = replace_if_exists

        response = requests.post(
            f"{API_HOST}/llm/unlearning-data/",
            json=payload,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_json = response.json()
        self.id = response_json.get("id")
        if self.id is None:
            raise ValueError("Server response did not include an unlearning data ID")
        return self.id

    @staticmethod
    def get_by_id(data_id: int) -> LlmUnlearningData:
        """Retrieve unlearning data from the Hirundo server by ID."""

        response = requests.get(
            f"{API_HOST}/llm/unlearning-data/{data_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        return LlmUnlearningData(**response.json())

    @staticmethod
    def list_datasets(organization_id: int | None = None) -> list[LlmUnlearningData]:
        """List unlearning datasets for the authenticated organization."""

        response = requests.get(
            f"{API_HOST}/llm/unlearning-data/",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
            params={"organization_id": organization_id} if organization_id else None,
        )
        raise_for_status_with_reason(response)
        return [LlmUnlearningData(**data_json) for data_json in response.json()]

    @staticmethod
    def delete_by_id(data_id: int) -> None:
        """Delete unlearning data by ID."""

        response = requests.delete(
            f"{API_HOST}/llm/unlearning-data/{data_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)

    def delete(self) -> None:
        """Delete this unlearning data resource."""

        if self.id is None:
            raise ValueError("Unlearning data must have an ID before it can be deleted")
        self.delete_by_id(self.id)


class LlmUnlearningRun(BaseModel):
    """A representation of an LLM unlearning run."""

    run_id: str
    status: str | None = None

    model_config = ConfigDict(extra="allow")

    @staticmethod
    def launch(model_id: int, run_info: BiasRunInfo) -> LlmUnlearningRun:
        """Launch an LLM unlearning run for a model."""

        if model_id is None:
            raise ValueError("Model ID is required to launch an unlearning run")
        payload = run_info.to_payload()
        response = requests.post(
            f"{API_HOST}/llm/unlearning/run/{model_id}",
            json=payload if payload else None,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        response_json = response.json()
        if "run_id" not in response_json:
            raise ValueError("Server response did not include a run ID")
        return LlmUnlearningRun(**response_json)

    @staticmethod
    def get(run_id: str) -> LlmUnlearningRun:
        """Retrieve details for an unlearning run by ID."""

        response = requests.get(
            f"{API_HOST}/llm/unlearning/run/{run_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        return LlmUnlearningRun(**response.json())
