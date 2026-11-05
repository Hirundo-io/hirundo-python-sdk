from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSourceType(str, Enum):
    HUGGINGFACE_TRANSFORMERS = "huggingface_transformers"
    LOCAL_TRANSFORMERS = "local_transformers"


class HuggingFaceTransformersModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    revision: str | None = None
    code_revision: str | None = None
    model_name: str
    token: str | None = None


class HuggingFaceTransformersModelOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    type: Literal[ModelSourceType.HUGGINGFACE_TRANSFORMERS] = (
        ModelSourceType.HUGGINGFACE_TRANSFORMERS
    )
    model_name: str


class LocalTransformersModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))

    type: Literal[ModelSourceType.LOCAL_TRANSFORMERS] = (
        ModelSourceType.LOCAL_TRANSFORMERS
    )
    local_path: str
    revision: str | None = None
    code_revision: str | None = None
    parameter_count: int | None = Field(default=None, gt=0)
    trust_remote_code: bool = True


LlmSources = HuggingFaceTransformersModel | LocalTransformersModel
LlmSourcesOutput = HuggingFaceTransformersModelOutput | LocalTransformersModel
