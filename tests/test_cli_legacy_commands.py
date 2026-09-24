from unittest.mock import patch

import pytest
import typer
from hirundo._env import EnvLocation
from hirundo.cli import KeyStorage, RunType, check_run, setup
from hirundo.cli_unlearning import unlearning_run


def test_setup_uses_one_environment_location_for_both_values() -> None:
    with (
        patch(
            "hirundo.cli._preferred_env_location",
            return_value=EnvLocation.HOME,
        ),
        patch(
            "hirundo.cli._save_api_host",
            return_value="https://app.hirundo.io",
        ) as save_api_host_mock,
        patch("hirundo.cli._save_api_key") as save_api_key_mock,
    ):
        setup("secret-key", "https://app.hirundo.io")

    save_api_host_mock.assert_called_once_with(
        "https://app.hirundo.io", EnvLocation.HOME
    )
    save_api_key_mock.assert_called_once_with(
        "secret-key",
        "https://app.hirundo.io",
        KeyStorage.AUTO,
        EnvLocation.HOME,
    )


def test_legacy_check_allows_dotted_external_run_id() -> None:
    with patch("hirundo.external_eval.ExternalEval") as external_eval_mock:
        external_eval_mock.check_run_by_id.return_value = None
        check_run("inspect.run-1", RunType.EXTERNAL_EVALUATION)

    external_eval_mock.check_run_by_id.assert_called_once_with("inspect.run-1")


@pytest.mark.parametrize(
    ("security", "refusal", "expected_type"),
    [(True, False, "SECURITY"), (False, True, "REFUSAL")],
)
def test_unlearning_run_supports_flag_behaviors(
    security: bool, refusal: bool, expected_type: str
) -> None:
    with patch("hirundo.unlearning_llm.LlmUnlearningRun") as unlearning_run_mock:
        unlearning_run_mock.get_capabilities.return_value.refusal_unlearning_enabled = (
            True
        )
        unlearning_run_mock.launch.return_value = "behavior-run"
        unlearning_run(42, security=security, refusal=refusal, wait=False)

    run_info = unlearning_run_mock.launch.call_args.args[1]
    assert run_info.target_behaviors[0].type == expected_type


def test_unlearning_run_rejects_refusal_when_capability_is_disabled() -> None:
    with patch("hirundo.unlearning_llm.LlmUnlearningRun") as unlearning_run_mock:
        unlearning_run_mock.get_capabilities.return_value.refusal_unlearning_enabled = (
            False
        )

        with pytest.raises(typer.BadParameter, match="not enabled"):
            unlearning_run(42, refusal=True, wait=False)

    unlearning_run_mock.launch.assert_not_called()
