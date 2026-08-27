import enum
import os
import warnings
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class EnvLocation(enum.Enum):
    DOTENV = find_dotenv(".env")
    HOME = Path.home() / ".hirundo.conf"


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


API_HOST = _get_env_with_deprecation(
    "HIRUNDO_API_HOST", "API_HOST", default="https://api.hirundo.io"
)
API_KEY = _get_env_with_deprecation("HIRUNDO_API_KEY", "API_KEY")


def check_api_key():
    if not API_KEY:
        raise ValueError(
            "HIRUNDO_API_KEY is not set. Please run `hirundo setup` to set the API key"
        )
