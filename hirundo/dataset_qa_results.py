import typing
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import TypeAliasType

from hirundo._dataframe import has_pandas, has_polars

DataFrameType = TypeAliasType("DataFrameType", None)

if has_pandas:
    from hirundo._dataframe import pd

    DataFrameType = TypeAliasType("DataFrameType", typing.Union[pd.DataFrame, None])
if has_polars:
    from hirundo._dataframe import pl

    DataFrameType = TypeAliasType("DataFrameType", typing.Union[pl.DataFrame, None])


T = typing.TypeVar("T")


class DatasetQAResults(BaseModel, typing.Generic[T]):
    model_config = {"arbitrary_types_allowed": True}

    cached_zip_path: Path
    """
    The path to the cached zip file of the results
    """
    suspects: T
    """
    A polars/pandas DataFrame containing the results of the data QA run
    """
    object_suspects: typing.Optional[T]
    """
    A polars/pandas DataFrame containing the object-level results of the data QA run
    """
    warnings_and_errors: T
    """
    A polars/pandas DataFrame containing the warnings and errors of the data QA run
    """
