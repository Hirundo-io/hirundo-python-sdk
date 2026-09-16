from unittest.mock import patch

from hirundo._env import _resolve_api_key


def test_environment_api_key_takes_precedence_over_keyring() -> None:
    with patch("hirundo._env.load_api_key_from_keyring") as load_keyring_mock:
        api_key = _resolve_api_key(
            "https://api.hirundo.io",
            api_key_from_environment="environment-secret",
            legacy_api_key_from_environment=None,
        )

    assert api_key == "environment-secret"
    load_keyring_mock.assert_not_called()


def test_keyring_api_key_takes_precedence_over_configuration_file() -> None:
    with (
        patch(
            "hirundo._env.load_api_key_from_keyring",
            return_value="keyring-secret",
        ),
        patch("hirundo._env._get_env_with_deprecation") as load_file_mock,
    ):
        api_key = _resolve_api_key(
            "https://api.hirundo.io",
            api_key_from_environment=None,
            legacy_api_key_from_environment=None,
        )

    assert api_key == "keyring-secret"
    load_file_mock.assert_not_called()


def test_configuration_file_is_used_when_keyring_is_unavailable() -> None:
    with (
        patch("hirundo._env.load_api_key_from_keyring", return_value=None),
        patch(
            "hirundo._env._get_env_with_deprecation",
            return_value="file-secret",
        ) as load_file_mock,
    ):
        api_key = _resolve_api_key(
            "https://api.hirundo.io",
            api_key_from_environment=None,
            legacy_api_key_from_environment=None,
        )

    assert api_key == "file-secret"
    load_file_mock.assert_called_once_with("HIRUNDO_API_KEY", "API_KEY")
