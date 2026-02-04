import typing
from pathlib import Path

from pydantic import BaseModel

T = typing.TypeVar("T")


class LlmBehaviorEvalResults(BaseModel, typing.Generic[T]):
    model_config = {"arbitrary_types_allowed": True}

    cached_zip_path: Path
    """
    The path to the cached zip file of the results
    """
    model_name: str | None = None
    """
    The model name used to locate results in the zip file
    """
    summary_brief: T
    """
    A polars/pandas DataFrame containing the summary_brief CSV
    """
    summary_full: T
    """
    A polars/pandas DataFrame containing the summary_full CSV
    """
