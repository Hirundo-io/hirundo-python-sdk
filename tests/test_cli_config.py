import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import dotenv_values
from hirundo.cli import _upsert_env, app
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


def test_upsert_env_rejects_symbolic_link(tmp_path: Path) -> None:
    target_path = tmp_path / "target"
    target_path.write_text("OTHER_SETTING='preserved'\n")
    dotenv_path = tmp_path / ".env"
    dotenv_path.symlink_to(target_path)

    with pytest.raises(ValueError, match="must be a regular file"):
        _upsert_env(dotenv_path, "HIRUNDO_API_KEY", "secret")

    assert dotenv_path.is_symlink()
    assert target_path.read_text() == "OTHER_SETTING='preserved'\n"


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
    save_api_key_mock.assert_called_once_with("secret-value")
