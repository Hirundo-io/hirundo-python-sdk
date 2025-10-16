from typing import Annotated

from pydantic import StringConstraints, UrlConstraints
from pydantic_core import Url

from hirundo.dataset_enum import StorageTypes

S3BucketUrl = Annotated[
    str,
    StringConstraints(
        min_length=8,
        max_length=1023,
        pattern=r"s3?://[a-z0-9.-]{3,64}[/]?",  # Only allow real S3 bucket URLs
    ),
]

StorageConfigName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9-_]+$",
    ),
]

STORAGE_PATTERNS: dict[StorageTypes, str] = {
    StorageTypes.S3: r"^s3:\/\/[a-z0-9\.\-]{3,63}/[a-zA-Z0-9!\-\/_\.\*'\(\)]+$",
    StorageTypes.GCP: r"^gs:\/\/([a-z0-9][a-z0-9_-]{1,61}[a-z0-9](\.[a-z0-9][a-z0-9_-]{1,61}[a-z0-9])*)\/[^\x00-\x1F\x7F-\x9F\r\n]*$",
}


LENGTH_CONSTRAINTS: dict[StorageTypes, dict] = {
    StorageTypes.S3: {"min_length": 8, "max_length": 1023, "bucket_max_length": None},
    StorageTypes.GCP: {"min_length": 8, "max_length": 1023, "bucket_max_length": 222},
}

RepoUrl = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=[
            "ssh",
            "https",
            "http",
        ]
    ),
]
HirundoUrl = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=[
            "file",
            "https",
            "http",
            "s3",
            "gs",
            "ssh",
            "hf",
        ]
    ),
]
