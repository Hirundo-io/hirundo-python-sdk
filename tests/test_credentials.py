from unittest.mock import Mock, patch

import pytest
from hirundo._credentials import (
    KEYRING_SERVICE,
    KeyringUnavailableError,
    load_api_key_from_keyring,
    save_api_key_to_keyring,
)


def test_load_api_key_from_keyring() -> None:
    backend = Mock(priority=5)
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
    backend = Mock(priority=5)
    with (
        patch("hirundo._credentials.keyring.get_keyring", return_value=backend),
        patch("hirundo._credentials.keyring.set_password") as set_password_mock,
    ):
        backend_name = save_api_key_to_keyring("https://api.hirundo.io", "secret")

    assert backend_name == "Mock"
    set_password_mock.assert_called_once_with(
        KEYRING_SERVICE, "https://api.hirundo.io", "secret"
    )


def test_save_api_key_fails_without_usable_backend() -> None:
    backend = Mock(priority=0)
    with patch("hirundo._credentials.keyring.get_keyring", return_value=backend):
        with pytest.raises(KeyringUnavailableError):
            save_api_key_to_keyring("https://api.hirundo.io", "secret")
