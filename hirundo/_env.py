import enum
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class EnvLocation(enum.Enum):
    DOTENV = find_dotenv(".env")
    HOME = Path.home() / ".hirundo.conf"


if os.path.exists(EnvLocation.DOTENV.value):
    load_dotenv(EnvLocation.DOTENV.value)
elif os.path.exists(EnvLocation.HOME.value):
    load_dotenv(EnvLocation.HOME.value)

API_HOST = os.getenv("API_HOST", "https://api.hirundo.io")
API_KEY = os.getenv("API_KEY")


def get_env_bool(variable_name: str, default: bool = False) -> bool:
    variable_value = os.getenv(variable_name)
    if variable_value is None:
        return default

    normalized_value = variable_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    return default


def check_api_key():
    if not API_KEY:
        raise ValueError(
            "API_KEY is not set. Please run `hirundo setup` to set the API key"
        )
