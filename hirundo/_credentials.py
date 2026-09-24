from urllib.parse import urlparse

import keyring
from keyring.backend import KeyringBackend
from keyring.backends.chainer import ChainerBackend
from keyring.errors import PasswordDeleteError

KEYRING_SERVICE = "hirundo-python-sdk"
_TRUSTED_BACKENDS = {
    ("keyring.backends.SecretService", "Keyring"),
    ("keyring.backends.Windows", "WinVaultKeyring"),
    ("keyring.backends.kwallet", "DBusKeyring"),
    ("keyring.backends.libsecret", "Keyring"),
    ("keyring.backends.macOS", "Keyring"),
}


class KeyringUnavailableError(RuntimeError):
    """Raised when the operating system has no usable credential store."""


def normalize_api_host(api_host: str) -> str:
    """Return the canonical origin used for API requests and keyring accounts.

    Args:
        api_host: API server address to validate and normalize.

    Returns:
        A canonical HTTP or HTTPS origin without a path, query, or fragment.
    """
    candidate = api_host.strip()
    if not candidate.lower().startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("API host must be a valid HTTP or HTTPS address.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("API host must not contain user credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("API host must not contain a query string or fragment.")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("API host contains an invalid port.") from error
    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port_suffix = f":{port}" if port is not None else ""
    return f"{parsed.scheme.lower()}://{hostname}{port_suffix}"


def _backend_is_trusted(backend: KeyringBackend) -> bool:
    backend_id = (type(backend).__module__, type(backend).__name__)
    if backend_id in _TRUSTED_BACKENDS:
        return True
    if not isinstance(backend, ChainerBackend):
        return False
    eligible_backends = [
        candidate for candidate in backend.backends if candidate.priority > 0
    ]
    return bool(eligible_backends) and all(
        _backend_is_trusted(candidate) for candidate in eligible_backends
    )


def _require_keyring_backend() -> KeyringBackend:
    try:
        backend = keyring.get_keyring()
        if backend.priority <= 0 or not _backend_is_trusted(backend):
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
        return keyring.get_password(KEYRING_SERVICE, normalize_api_host(api_host))
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
        keyring.set_password(KEYRING_SERVICE, normalize_api_host(api_host), api_key)
    except Exception as error:
        raise KeyringUnavailableError from error
    return type(backend).__name__


def delete_api_key_from_keyring(api_host: str) -> bool:
    """Delete the keyring credential for an API host when one exists.

    Args:
        api_host: API server address used to identify the credential.

    Returns:
        True when a credential was deleted, otherwise False.
    """
    try:
        backend = keyring.get_keyring()
        if backend.priority <= 0:
            return False
        keyring.delete_password(KEYRING_SERVICE, normalize_api_host(api_host))
    except PasswordDeleteError:
        return False
    except Exception as error:
        raise KeyringUnavailableError from error
    return True
