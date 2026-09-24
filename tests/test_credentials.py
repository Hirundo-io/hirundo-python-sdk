from unittest.mock import Mock, patch

import pytest
from hirundo._credentials import (
    KEYRING_SERVICE,
    KeyringUnavailableError,
    delete_api_key_from_keyring,
    load_api_key_from_keyring,
    normalize_api_host,
    save_api_key_to_keyring,
)

TrustedBackend = type("Keyring", (), {"priority": 5})
TrustedBackend.__module__ = "keyring.backends.macOS"
PlaintextBackend = type("PlaintextKeyring", (), {"priority": 5})
PlaintextBackend.__module__ = "keyrings.alt.file"


def test_load_api_key_from_keyring() -> None:
    backend = TrustedBackend()
    with (
        patch("hirundo._credentials.keyring.get_keyring", return_value=backend),
        patch(
            "hirundo._credentials.keyring.get_password",
            return_value="secret",
        ) as get_password_mock,
    ):
        api_key = load_api_key_from_keyring("https://api.hirundo.io")

    assert api_key == "secret"
    get_password_mock.assert_called_once_with(KEYRING_SERVICE, "https://api.hirundo.io")


def test_load_api_key_returns_none_without_usable_backend() -> None:
    backend = Mock(priority=0)
    with patch("hirundo._credentials.keyring.get_keyring", return_value=backend):
        assert load_api_key_from_keyring("https://api.hirundo.io") is None


def test_save_api_key_to_keyring() -> None:
    backend = TrustedBackend()
    with (
        patch("hirundo._credentials.keyring.get_keyring", return_value=backend),
        patch("hirundo._credentials.keyring.set_password") as set_password_mock,
    ):
        backend_name = save_api_key_to_keyring("https://api.hirundo.io", "secret")

    assert backend_name == "Keyring"
    set_password_mock.assert_called_once_with(
        KEYRING_SERVICE, "https://api.hirundo.io", "secret"
    )


def test_save_api_key_fails_without_usable_backend() -> None:
    backend = Mock(priority=0)
    with patch("hirundo._credentials.keyring.get_keyring", return_value=backend):
        with pytest.raises(KeyringUnavailableError):
            save_api_key_to_keyring("https://api.hirundo.io", "secret")


def test_positive_priority_plaintext_backend_is_rejected() -> None:
    backend = PlaintextBackend()
    with patch("hirundo._credentials.keyring.get_keyring", return_value=backend):
        with pytest.raises(KeyringUnavailableError):
            save_api_key_to_keyring("https://api.hirundo.io", "secret")


def test_api_host_is_canonicalized_for_keyring_account() -> None:
    backend = TrustedBackend()
    with (
        patch("hirundo._credentials.keyring.get_keyring", return_value=backend),
        patch("hirundo._credentials.keyring.set_password") as set_password_mock,
    ):
        save_api_key_to_keyring("HTTPS://API.HIRUNDO.IO/path/", "secret")

    set_password_mock.assert_called_once_with(
        KEYRING_SERVICE, "https://api.hirundo.io", "secret"
    )


@pytest.mark.parametrize(
    "api_host",
    [
        "https://user:secret@api.hirundo.io",
        "https://api.hirundo.io?token=secret",
        "https://api.hirundo.io#fragment",
    ],
)
def test_api_host_rejects_credential_bearing_components(api_host: str) -> None:
    with pytest.raises(ValueError):
        normalize_api_host(api_host)


def test_delete_api_key_from_keyring() -> None:
    backend = TrustedBackend()
    with (
        patch("hirundo._credentials.keyring.get_keyring", return_value=backend),
        patch("hirundo._credentials.keyring.delete_password") as delete_mock,
    ):
        assert delete_api_key_from_keyring("https://api.hirundo.io/") is True

    delete_mock.assert_called_once_with(KEYRING_SERVICE, "https://api.hirundo.io")
