from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


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
    type: Literal[ModelSourceType.LOCAL_TRANSFORMERS] = (
        ModelSourceType.LOCAL_TRANSFORMERS
    )
    revision: None = None
    code_revision: None = None
    local_path: str


LlmSources = HuggingFaceTransformersModel | LocalTransformersModel
LlmSourcesOutput = HuggingFaceTransformersModelOutput | LocalTransformersModel
