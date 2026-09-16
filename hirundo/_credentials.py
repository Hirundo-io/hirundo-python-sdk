import keyring
from keyring.backend import KeyringBackend

KEYRING_SERVICE = "hirundo-python-sdk"


class KeyringUnavailableError(RuntimeError):
    """Raised when the operating system has no usable credential store."""


def _require_keyring_backend() -> KeyringBackend:
    try:
        backend = keyring.get_keyring()
        if backend.priority <= 0:
            raise KeyringUnavailableError
    except Exception as error:
        raise KeyringUnavailableError from error
    return backend


def load_api_key_from_keyring(api_host: str) -> str | None:
    """Load an API key, returning None when the keyring cannot be used.

    Args:
        api_host: API server address used to identify the credential.

    Returns:
        The stored API key, or None when no key exists or the keyring is unavailable.
    """
    try:
        _require_keyring_backend()
        return keyring.get_password(KEYRING_SERVICE, api_host)
    except Exception:
        return None


def save_api_key_to_keyring(api_host: str, api_key: str) -> str:
    """Save an API key in the operating system's credential store.

    Args:
        api_host: API server address used to identify the credential.
        api_key: Secret API key to store.

    Returns:
        The selected keyring backend's display name.
    """
    backend = _require_keyring_backend()
    try:
        keyring.set_password(KEYRING_SERVICE, api_host, api_key)
    except Exception as error:
        raise KeyringUnavailableError from error
    return type(backend).__name__
