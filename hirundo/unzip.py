import typing
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, cast

from pydantic_core import Url

from hirundo._dataframe import (
    float32,
    has_pandas,
    has_polars,
    int32,
    pd,
    pl,
    string,
)
from hirundo._env import API_HOST
from hirundo._headers import _get_auth_headers
from hirundo._http import requests
from hirundo._timeouts import DOWNLOAD_READ_TIMEOUT
from hirundo.dataset_qa_results import (
    DataFrameType,
    DatasetQAResults,
)
from hirundo.logger import get_logger

ZIP_FILE_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB

Dtype = typing.Union[type[int32], type[float32], type[string]]


CUSTOMER_INTERCHANGE_DTYPES: Mapping[str, Dtype] = {
    "image_path": string,
    "label_path": string,
    "segments_mask_path": string,
    "segment_id": int32,
    "label": string,
    "bbox_id": string,
    "xmin": float32,
    "ymin": float32,
    "xmax": float32,
    "ymax": float32,
    "suspect_level": float32,  # If exists, must be one of the values in the enum below
    "suggested_label": string,
    "suggested_label_conf": float32,
    "status": string,
    # ⬆️ If exists, must be one of the following:
    # NO_LABELS/MISSING_IMAGE/INVALID_IMAGE/INVALID_BBOX/INVALID_BBOX_SIZE/INVALID_SEG/INVALID_SEG_SIZE
}

logger = get_logger(__name__)


def _clean_df_index(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Clean the index of a DataFrame in case it has unnamed columns.

    Args:
        df (DataFrame): DataFrame to clean

    Returns:
        Cleaned Pandas DataFrame
    """
    index_cols = sorted(
        [col for col in df.columns if col.startswith("Unnamed")], reverse=True
    )
    if len(index_cols) > 0:
        df.set_index(index_cols.pop(), inplace=True)
        df.rename_axis(index=None, columns=None, inplace=True)
        if len(index_cols) > 0:
            df.drop(columns=index_cols, inplace=True)

    return df


def load_df(
    file: "typing.Union[str, IO[bytes]]",
) -> "DataFrameType":
    """
    Load a DataFrame from a CSV file.

    Args:
        file_name: The name of the CSV file to load.
        dtypes: The data types of the columns in the DataFrame.

    Returns:
        The loaded DataFrame or `None` if neither Polars nor Pandas is available.
    """
    if has_polars:
        return pl.read_csv(file, schema_overrides=CUSTOMER_INTERCHANGE_DTYPES)
    elif has_pandas:
        if typing.TYPE_CHECKING:
            from pandas._typing import DtypeArg

        dtype = cast("DtypeArg", CUSTOMER_INTERCHANGE_DTYPES)
        #  ⬆️ Casting since CUSTOMER_INTERCHANGE_DTYPES is a Mapping[str, Dtype] in this case
        df = pd.read_csv(file, dtype=dtype)
        return cast("DataFrameType", _clean_df_index(df))
        #  ⬆️ Casting since the return type is pd.DataFrame, but this is what DataFrameType is in this case
    else:
        return None


def get_mislabel_suspect_filename(filenames: list[str]):
    mislabel_suspect_filename = "mislabel_suspects.csv"
    if mislabel_suspect_filename not in filenames:
        mislabel_suspect_filename = "image_mislabel_suspects.csv"
    if mislabel_suspect_filename not in filenames:
        mislabel_suspect_filename = "suspects.csv"
    if mislabel_suspect_filename not in filenames:
        raise ValueError(
            "None of mislabel_suspects.csv, image_mislabel_suspects.csv or suspects.csv were found in the zip file"
        )
    return mislabel_suspect_filename


def download_and_extract_zip(
    run_id: str, zip_url: str
) -> DatasetQAResults[DataFrameType]:
    """
    Download and extract the zip file from the given URL.

    Note: It will only extract the `mislabel_suspects.csv` (vision - classification)
    or `image_mislabel_suspects.csv` & `object_mislabel_suspects.csv` (vision - OD)
    or `suspects.csv` (STT)
    and `warnings_and_errors.csv` files from the zip file.

    Args:
        run_id: The ID of the dataset QA run.
        zip_url: The URL of the zip file to download.

    Returns:
        The dataset QA results object.
    """
    # Define the local file path
    cache_dir = Path.home() / ".hirundo" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = cache_dir / f"{run_id}.zip"

    headers = None
    if Url(zip_url).scheme == "file":
        zip_url = f"{API_HOST}/dataset-qa/run/local-download" + zip_url.replace(
            "file://", ""
        )
        headers = _get_auth_headers()
    # Stream the zip file download
    with requests.get(
        zip_url,
        headers=headers,
        timeout=DOWNLOAD_READ_TIMEOUT,
        stream=True,
    ) as r:
        r.raise_for_status()
        with open(zip_file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=ZIP_FILE_CHUNK_SIZE):
                f.write(chunk)
        logger.info(
            "Successfully downloaded the result zip file for run ID %s to %s",
            run_id,
            zip_file_path,
        )

        with zipfile.ZipFile(zip_file_path, "r") as z:
            # Extract suspects file
            suspects_df = None
            object_suspects_df = None
            warnings_and_errors_df = None

            filenames = []
            try:
                filenames = [file.filename for file in z.filelist]
            except Exception as e:
                logger.error("Failed to get filenames from ZIP", exc_info=e)

            try:
                mislabel_suspect_filename = get_mislabel_suspect_filename(filenames)
                with z.open(mislabel_suspect_filename) as suspects_file:
                    suspects_df = load_df(suspects_file)
                logger.debug(
                    "Successfully loaded mislabel suspects into DataFrame for run ID %s",
                    run_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to load mislabel suspects into DataFrame", exc_info=e
                )

            object_mislabel_suspects_filename = "object_mislabel_suspects.csv"
            if object_mislabel_suspects_filename in filenames:
                try:
                    with z.open(
                        object_mislabel_suspects_filename
                    ) as object_suspects_file:
                        object_suspects_df = load_df(object_suspects_file)
                    logger.debug(
                        "Successfully loaded object mislabel suspects into DataFrame for run ID %s",
                        run_id,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load object mislabel suspects into DataFrame",
                        exc_info=e,
                    )

            try:
                # Extract warnings_and_errors file
                with z.open("warnings_and_errors.csv") as warnings_file:
                    warnings_and_errors_df = load_df(warnings_file)
                logger.debug(
                    "Successfully loaded warnings and errors into DataFrame for run ID %s",
                    run_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to load warnings and errors into DataFrame", exc_info=e
                )

            return DatasetQAResults[DataFrameType](
                cached_zip_path=zip_file_path,
                suspects=suspects_df,
                object_suspects=object_suspects_df,
                warnings_and_errors=warnings_and_errors_df,
            )


def load_from_zip(
    zip_path: Path, file_name: str
) -> "typing.Union[pd.DataFrame, pl.DataFrame, None]":
    """
    Load a given file from a given zip file.

    Args:
        zip_path: The path to the zip file.
        file_name: The name of the file to load.

    Returns:
        The loaded DataFrame or `None` if neither Polars nor Pandas is available.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        try:
            with z.open(file_name) as file:
                return load_df(file)
        except Exception as e:
            logger.error("Failed to load %s from zip file", file_name, exc_info=e)
    return None
