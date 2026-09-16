import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from dotenv import dotenv_values
from hirundo._credentials import KeyringUnavailableError
from hirundo._env import EnvLocation
from hirundo.cli import (
    API_HOST,
    KeyStorage,
    _rewrite_env,
    _save_api_key,
    _upsert_env,
    app,
    setup,
)
from typer.testing import CliRunner


def test_upsert_env_creates_private_file_and_quotes_value(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".hirundo.conf"

    _upsert_env(dotenv_path, "HIRUNDO_API_KEY", "secret with 'quotes'")

    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600
    assert dotenv_values(dotenv_path)["HIRUNDO_API_KEY"] == "secret with 'quotes'"


def test_upsert_env_tightens_existing_file_permissions(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OTHER_SETTING='preserved'\n")
    dotenv_path.chmod(0o644)

    _upsert_env(dotenv_path, "HIRUNDO_API_KEY", "secret")

    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600
    assert dotenv_values(dotenv_path) == {
        "OTHER_SETTING": "preserved",
        "HIRUNDO_API_KEY": "secret",
    }


def test_upsert_env_updates_symbolic_link_target_securely(tmp_path: Path) -> None:
    target_path = tmp_path / "target"
    target_path.write_text("OTHER_SETTING='preserved'\n")
    dotenv_path = tmp_path / ".env"
    dotenv_path.symlink_to(target_path)

    _upsert_env(dotenv_path, "HIRUNDO_API_KEY", "secret")

    assert dotenv_path.is_symlink()
    assert dotenv_values(target_path) == {
        "OTHER_SETTING": "preserved",
        "HIRUNDO_API_KEY": "secret",
    }
    assert stat.S_IMODE(target_path.stat().st_mode) == 0o600


def test_rewrite_env_rejects_destination_changed_during_update(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("ORIGINAL='value'\n")

    def replace_destination(_: Path) -> None:
        dotenv_path.unlink()
        dotenv_path.write_text("ATTACKER='value'\n")

    with pytest.raises(RuntimeError, match="changed while updating"):
        _rewrite_env(dotenv_path, replace_destination)

    assert dotenv_values(dotenv_path) == {"ATTACKER": "value"}


@pytest.mark.parametrize("invalid_character", ["\r", "\n", "\0"])
def test_upsert_env_rejects_unsafe_api_key_characters(
    tmp_path: Path, invalid_character: str
) -> None:
    dotenv_path = tmp_path / ".hirundo.conf"

    with pytest.raises(ValueError, match="must not contain"):
        _upsert_env(
            dotenv_path,
            "HIRUNDO_API_KEY",
            f"secret{invalid_character}injected=value",
        )

    assert not dotenv_path.exists()


def test_set_api_key_masks_interactive_input() -> None:
    runner = CliRunner()

    with patch("hirundo.cli._save_api_key") as save_api_key_mock:
        result = runner.invoke(app, ["set-api-key"], input="secret-value\n")

    assert result.exit_code == 0
    assert "secret-value" not in result.output
    save_api_key_mock.assert_called_once_with("secret-value", API_HOST, KeyStorage.AUTO)


def test_setup_validates_api_key_before_writing_host() -> None:
    with (
        patch("hirundo.cli._save_api_host") as save_api_host_mock,
        patch("hirundo.cli._save_api_key") as save_api_key_mock,
        pytest.raises(ValueError, match="must not contain"),
    ):
        setup("bad\nkey", "https://api.hirundo.io")

    save_api_host_mock.assert_not_called()
    save_api_key_mock.assert_not_called()


def test_save_api_key_prefers_keyring_and_removes_plaintext_key(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HIRUNDO_API_KEY='old-secret'\nOTHER_SETTING='preserved'\n")
    location = EnvLocation.DOTENV

    with (
        patch.object(location, "_value_", str(dotenv_path)),
        patch(
            "hirundo.cli.save_api_key_to_keyring",
            return_value="TestKeyring",
        ) as save_to_keyring_mock,
    ):
        _save_api_key(
            "new-secret",
            "https://api.hirundo.io",
            KeyStorage.AUTO,
            location,
        )

    save_to_keyring_mock.assert_called_once_with("https://api.hirundo.io", "new-secret")
    assert dotenv_values(dotenv_path) == {"OTHER_SETTING": "preserved"}
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600


def test_save_api_key_auto_falls_back_to_private_file(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".hirundo.conf"
    location = EnvLocation.HOME

    with (
        patch.object(location, "_value_", dotenv_path),
        patch(
            "hirundo.cli.save_api_key_to_keyring",
            side_effect=KeyringUnavailableError,
        ),
    ):
        _save_api_key(
            "secret",
            "https://api.hirundo.io",
            KeyStorage.AUTO,
            location,
        )

    assert dotenv_values(dotenv_path)["HIRUNDO_API_KEY"] == "secret"
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600


def test_save_api_key_explicit_file_does_not_access_keyring(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".hirundo.conf"
    location = EnvLocation.HOME

    with (
        patch.object(location, "_value_", dotenv_path),
        patch("hirundo.cli.save_api_key_to_keyring") as save_to_keyring_mock,
    ):
        _save_api_key(
            "secret",
            "https://api.hirundo.io",
            KeyStorage.FILE,
            location,
        )

    save_to_keyring_mock.assert_not_called()
    assert dotenv_values(dotenv_path)["HIRUNDO_API_KEY"] == "secret"


def test_save_api_key_explicit_keyring_fails_closed() -> None:
    with patch(
        "hirundo.cli.save_api_key_to_keyring",
        side_effect=KeyringUnavailableError,
    ):
        with pytest.raises(
            typer.BadParameter,
            match="No usable operating-system keyring is available",
        ):
            _save_api_key(
                "secret",
                "https://api.hirundo.io",
                KeyStorage.KEYRING,
            )
