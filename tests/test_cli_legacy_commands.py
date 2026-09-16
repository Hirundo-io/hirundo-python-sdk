from unittest.mock import patch

from hirundo._env import EnvLocation
from hirundo.cli import RunType, check_run, setup


def test_setup_uses_one_environment_location_for_both_values() -> None:
    with (
        patch(
            "hirundo.cli._preferred_env_location",
            return_value=EnvLocation.HOME,
        ),
        patch("hirundo.cli._save_api_host") as save_api_host_mock,
        patch("hirundo.cli._save_api_key") as save_api_key_mock,
    ):
        setup("secret-key", "https://app.hirundo.io")

    save_api_host_mock.assert_called_once_with(
        "https://app.hirundo.io", EnvLocation.HOME
    )
    save_api_key_mock.assert_called_once_with("secret-key", EnvLocation.HOME)


def test_legacy_check_allows_dotted_external_run_id() -> None:
    with patch("hirundo.external_eval.ExternalEval") as external_eval_mock:
        external_eval_mock.check_run_by_id.return_value = None
        check_run("inspect.run-1", RunType.EXTERNAL_EVALUATION)

    external_eval_mock.check_run_by_id.assert_called_once_with("inspect.run-1")
