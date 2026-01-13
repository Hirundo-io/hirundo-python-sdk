import datetime
import typing
from enum import Enum

from pydantic import BaseModel
from typing_extensions import Literal

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo.logger import get_logger

logger = get_logger(__name__)


class ModelSourceType(str, Enum):
    HUGGINGFACE_TRANSFORMERS = "huggingface_transformers"
    LOCAL_TRANSFORMERS = "local_transformers"


class HuggingFaceTransformersModel(BaseModel):
    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    revision: typing.Optional[str] = None
    code_revision: typing.Optional[str] = None
    model_name: str
    token: typing.Optional[str] = None


class HuggingFaceTransformersModelOutput(BaseModel):
    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    model_name: str


class LocalTransformersModel(BaseModel):
    type: Literal[ModelSourceType.LOCAL_TRANSFORMERS] = (
        ModelSourceType.LOCAL_TRANSFORMERS
    )
    revision: None = None
    code_revision: None = None
    local_path: str


LlmSources = typing.Union[HuggingFaceTransformersModel, LocalTransformersModel]
LlmSourcesOutput = typing.Union[
    HuggingFaceTransformersModelOutput, LocalTransformersModel
]


class LlmModel(BaseModel):
    id: typing.Optional[int] = None
    organization_id: typing.Optional[int] = None
    model_name: str
    model_source: LlmSources
    archive_existing_runs: bool = True

    def create(
        self,
        replace_if_exists: bool = False,
    ) -> int:
        llm_model_response = requests.post(
            f"{API_HOST}/unlearning-llm/llm/",
            json={
                **self.model_dump(mode="json"),
                "replace_if_exists": replace_if_exists,
            },
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        llm_model_id = llm_model_response.json()["id"]
        self.id = llm_model_id
        return llm_model_id

    @staticmethod
    def get_by_id(llm_model_id: int) -> "LlmModelOut":
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/{llm_model_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        return LlmModelOut.from_response(llm_model_response.json())

    @staticmethod
    def get_by_name(llm_model_name: str) -> "LlmModelOut":
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/by-name/{llm_model_name}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        return LlmModelOut.from_response(llm_model_response.json())

    @staticmethod
    def list(organization_id: typing.Optional[int] = None) -> list["LlmModelOut"]:
        params = {}
        if organization_id is not None:
            params["model_organization_id"] = organization_id
        llm_model_response = requests.get(
            f"{API_HOST}/unlearning-llm/llm/",
            params=params,
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        llm_model_json = llm_model_response.json()
        return [LlmModelOut.from_response(llm_model) for llm_model in llm_model_json]

    @staticmethod
    def delete_by_id(llm_model_id: int) -> None:
        llm_model_response = requests.delete(
            f"{API_HOST}/unlearning-llm/llm/{llm_model_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        logger.info("Deleted LLM model with ID: %s", llm_model_id)

    def delete(self) -> None:
        if not self.id:
            raise ValueError("No LLM model has been created")
        self.delete_by_id(self.id)

    def update(
        self,
        model_name: typing.Optional[str] = None,
        model_source: typing.Optional[LlmSources] = None,
        archive_existing_runs: typing.Optional[bool] = None,
    ) -> None:
        if not self.id:
            raise ValueError("No LLM model has been created")
        payload: dict[str, typing.Any] = {
            "model_name": model_name,
            "model_source": model_source.model_dump(mode="json")
            if model_source
            else None,
            "archive_existing_runs": archive_existing_runs,
            "organization_id": self.organization_id,
        }
        llm_model_response = requests.put(
            f"{API_HOST}/unlearning-llm/llm/{self.id}",
            json=payload,
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(llm_model_response)
        if model_name is not None:
            self.model_name = model_name
        if model_source is not None:
            self.model_source = model_source
        if archive_existing_runs is not None:
            self.archive_existing_runs = archive_existing_runs


class LlmModelOut(BaseModel):
    id: int
    organization_id: int
    creator_id: int
    creator_name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    model_name: str
    model_source: LlmSourcesOutput

    @staticmethod
    def from_response(response_payload: dict[str, typing.Any]) -> "LlmModelOut":
        return LlmModelOut(
            id=response_payload["id"],
            organization_id=response_payload["organization_id"],
            creator_id=response_payload["creator_id"],
            creator_name=response_payload["creator_name"],
            created_at=response_payload["created_at"],
            updated_at=response_payload["updated_at"],
            model_name=response_payload["model_name"],
            model_source=response_payload["model_source"],
        )


class DatasetType(str, Enum):
    NORMAL = "normal"
    BIAS = "bias"
    UNBIAS = "unbias"


class UnlearningLlmAdvancedOptions(BaseModel):
    max_tokens_for_model: typing.Optional[typing.Union[dict[DatasetType, int], int]] = (
        None
    )


class BiasType(str, Enum):
    ALL = "ALL"
    RACE = "RACE"
    NATIONALITY = "NATIONALITY"
    GENDER = "GENDER"
    PHYSICAL_APPEARANCE = "PHYSICAL_APPEARANCE"
    RELIGION = "RELIGION"
    AGE = "AGE"


class UtilityType(str, Enum):
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"


class DefaultUtility(BaseModel):
    utility_type: Literal[UtilityType.DEFAULT] = UtilityType.DEFAULT


class HirundoCSVDataset(BaseModel):
    type: Literal["HirundoCSV"] = "HirundoCSV"
    csv_url: str


class HuggingFaceDataset(BaseModel):
    type: Literal["HuggingFaceDataset"] = "HuggingFaceDataset"
    hugging_face_dataset_name: str


CustomDataset = typing.Union[HirundoCSVDataset, HuggingFaceDataset]


class CustomUtility(BaseModel):
    utility_type: Literal[UtilityType.CUSTOM] = UtilityType.CUSTOM
    dataset: CustomDataset


class BiasBehavior(BaseModel):
    type: Literal["BIAS"] = "BIAS"
    bias_type: BiasType


class HallucinationType(str, Enum):
    GENERAL = "GENERAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    DEFENSE = "DEFENSE"


class HallucinationBehavior(BaseModel):
    type: Literal["HALLUCINATION"] = "HALLUCINATION"
    hallucination_type: HallucinationType


class SecurityBehavior(BaseModel):
    type: Literal["SECURITY"] = "SECURITY"


class CustomBehavior(BaseModel):
    type: Literal["CUSTOM"] = "CUSTOM"
    biased_dataset: CustomDataset
    unbiased_dataset: CustomDataset


TargetBehavior = typing.Union[
    BiasBehavior,
    HallucinationBehavior,
    SecurityBehavior,
    CustomBehavior,
]

TargetUtility = typing.Union[DefaultUtility, CustomUtility]


class LlmRunInfo(BaseModel):
    organization_id: typing.Optional[int] = None
    name: typing.Optional[str] = None
    model_id: typing.Optional[int] = None
    target_behaviors: list[TargetBehavior]
    target_utilities: list[TargetUtility]
    advanced_options: typing.Optional[UnlearningLlmAdvancedOptions] = None


class BiasRunInfo(BaseModel):
    bias_type: BiasType
    organization_id: typing.Optional[int] = None
    name: typing.Optional[str] = None
    target_utilities: typing.Optional[list[TargetUtility]] = None
    advanced_options: typing.Optional[UnlearningLlmAdvancedOptions] = None

    def to_run_info(self) -> LlmRunInfo:
        default_utilities = (
            self.target_utilities
            if self.target_utilities is not None
            else [DefaultUtility()]
        )
        return LlmRunInfo(
            organization_id=self.organization_id,
            name=self.name,
            target_behaviors=[BiasBehavior(bias_type=self.bias_type)],
            target_utilities=default_utilities,
            advanced_options=self.advanced_options,
        )


class LlmUnlearningRun:
    @staticmethod
    def launch(model_id: int, run_info: typing.Union[LlmRunInfo, BiasRunInfo]) -> str:
        resolved_run_info = (
            run_info.to_run_info() if isinstance(run_info, BiasRunInfo) else run_info
        )
        run_response = requests.post(
            f"{API_HOST}/unlearning-llm/run/{model_id}",
            json=resolved_run_info.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        run_response_json = run_response.json() if run_response.content else {}
        if isinstance(run_response_json, str):
            return run_response_json
        run_id = (
            run_response_json.get("run_id")
            or run_response_json.get("hir_run_id")
            or run_response_json.get("id")
        )
        if not run_id:
            raise ValueError("No run ID returned from launch request")
        return run_id

    @staticmethod
    def cancel(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/cancel/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def rename(run_id: str, new_name: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/rename/{run_id}",
            json={"new_name": new_name},
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def archive(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/archive/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def restore(run_id: str) -> None:
        run_response = requests.patch(
            f"{API_HOST}/unlearning-llm/run/restore/{run_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)

    @staticmethod
    def list(
        organization_id: typing.Optional[int] = None,
        archived: bool = False,
    ) -> list[dict[str, object]]:
        params: dict[str, bool | int] = {"archived": archived}
        if organization_id is not None:
            params["unlearning_organization_id"] = organization_id
        run_response = requests.get(
            f"{API_HOST}/unlearning-llm/run/list",
            params=params,
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(run_response)
        response_json = run_response.json()
        if isinstance(response_json, list):
            return response_json
        return [response_json]
