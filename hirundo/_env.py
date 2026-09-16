import enum
import os
import warnings
from pathlib import Path
from typing import cast

from dotenv import find_dotenv, load_dotenv

from hirundo._credentials import load_api_key_from_keyring


class EnvLocation(enum.Enum):
    DOTENV = find_dotenv(".env")
    HOME = Path.home() / ".hirundo.conf"


api_key_from_environment = os.getenv("HIRUNDO_API_KEY")
legacy_api_key_from_environment = os.getenv("API_KEY")

if os.path.exists(EnvLocation.DOTENV.value):
    load_dotenv(EnvLocation.DOTENV.value)
elif os.path.exists(EnvLocation.HOME.value):
    load_dotenv(EnvLocation.HOME.value)


def _get_env_with_deprecation(new_name: str, old_name: str, default: str | None = None):
    new_value = os.getenv(new_name)
    if new_value is not None:
        return new_value

    old_value = os.getenv(old_name)
    if old_value is not None:
        warnings.warn(
            (
                f"Environment variable '{old_name}' is deprecated and will be removed "
                f"in a future release. Use '{new_name}' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return old_value

    return default


def _resolve_api_key(
    api_host: str,
    api_key_from_environment: str | None,
    legacy_api_key_from_environment: str | None,
) -> str | None:
    if api_key_from_environment is not None:
        return api_key_from_environment
    if legacy_api_key_from_environment is not None:
        warnings.warn(
            (
                "Environment variable 'API_KEY' is deprecated and will be removed "
                "in a future release. Use 'HIRUNDO_API_KEY' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_api_key_from_environment

    keyring_api_key = load_api_key_from_keyring(api_host)
    if keyring_api_key is not None:
        return keyring_api_key
    return _get_env_with_deprecation("HIRUNDO_API_KEY", "API_KEY")


API_HOST = cast(
    "str",
    _get_env_with_deprecation(
        "HIRUNDO_API_HOST", "API_HOST", default="https://api.hirundo.io"
    ),
)
API_KEY = _resolve_api_key(
    API_HOST, api_key_from_environment, legacy_api_key_from_environment
)
EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("HIRUNDO_EXTERNAL_EVAL_ALLOWED_DOWNLOAD_ORIGINS", "").split(
        ","
    )
    if origin.strip()
)


def check_api_key() -> None:
    if not API_KEY:
        raise ValueError(
            "Hirundo API key is not configured. Set HIRUNDO_API_KEY or run "
            "`hirundo setup`."
        )
