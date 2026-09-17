import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from hirundo import _cli_common
from hirundo._cli_common import OutputFormat, run_payload, set_output_format
from hirundo._hirundo_error import HirundoError
from hirundo._http import requests
from hirundo._run_status import RunStatus
from hirundo.cli import app
from typer import _click as click
from typer.core import TyperGroup
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_output_format():
    set_output_format(OutputFormat.text)
    yield
    set_output_format(OutputFormat.text)


class TestRunPayload:
    def test_includes_cached_zip_path_when_present(self):
        results = MagicMock(cached_zip_path="cache/run.zip")
        assert run_payload("r1", results) == {
            "run_id": "r1",
            "cached_zip_path": "cache/run.zip",
        }

    def test_none_when_results_missing_attr(self):
        # unlearning returns a raw dict with no cached_zip_path
        assert run_payload("r1", {"iteration": 1}) == {
            "run_id": "r1",
            "cached_zip_path": None,
        }

    def test_none_when_no_results(self):
        assert run_payload("r1") == {"run_id": "r1", "cached_zip_path": None}


class TestJsonOutput:
    def test_run_no_wait_emits_clean_json(self):
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.return_value = "run-abc"
            result = runner.invoke(
                app, ["dataset-qa", "run", "42", "--no-wait", "-o", "json"]
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == {
            "run_id": "run-abc",
            "cached_zip_path": None,
        }

    def test_unlearning_bias_uses_current_behavior_model(self):
        with patch("hirundo.unlearning_llm.LlmUnlearningRun") as unlearning_run_mock:
            unlearning_run_mock.launch.return_value = "run-bias"
            result = runner.invoke(
                app,
                [
                    "unlearning",
                    "run",
                    "42",
                    "--bias",
                    "--no-wait",
                    "-o",
                    "json",
                ],
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == {
            "run_id": "run-bias",
            "cached_zip_path": None,
        }
        run_info = unlearning_run_mock.launch.call_args.args[1]
        assert run_info.target_behaviors[0].type == "BIAS"

    @pytest.mark.parametrize(
        ("behavior_flag", "expected_type"),
        [("--security", "SECURITY"), ("--refusal", "REFUSAL")],
    )
    def test_unlearning_supports_flag_behaviors(self, behavior_flag, expected_type):
        with patch("hirundo.unlearning_llm.LlmUnlearningRun") as unlearning_run_mock:
            unlearning_run_mock.launch.return_value = "run-behavior"
            result = runner.invoke(
                app,
                [
                    "unlearning",
                    "run",
                    "42",
                    behavior_flag,
                    "--no-wait",
                    "-o",
                    "json",
                ],
            )
        assert result.exit_code == 0
        run_info = unlearning_run_mock.launch.call_args.args[1]
        assert run_info.target_behaviors[0].type == expected_type

    def test_list_emits_json_array(self):
        run_record = MagicMock(
            name="dataset", run_id="r1", status="COMPLETED", run_args=None
        )
        run_record.name = "dataset"
        run_record.created_at.isoformat.return_value = "2026-05-31T00:00:00"
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.list_runs.return_value = [run_record]
            result = runner.invoke(app, ["dataset-qa", "list", "-o", "json"])
        assert result.exit_code == 0
        assert json.loads(result.stdout) == [
            {
                "dataset_name": "dataset",
                "run_id": "r1",
                "status": "COMPLETED",
                "created_at": "2026-05-31T00:00:00",
                "run_args": None,
            }
        ]
        assert result.stderr == ""

    def test_list_serializes_status_enum_value(self):
        run_record = MagicMock(
            name="dataset", run_id="r1", status=RunStatus.SUCCESS, run_args=None
        )
        run_record.name = "dataset"
        run_record.created_at.isoformat.return_value = "2026-05-31T00:00:00"
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.list_runs.return_value = [run_record]
            result = runner.invoke(app, ["dataset-qa", "list", "-o", "json"])
        assert json.loads(result.stdout)[0]["status"] == "SUCCESS"

    def test_check_emits_clean_json(self):
        check_results = MagicMock(cached_zip_path="cache/run-abc.zip")
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.check_run_by_id.return_value = check_results
            result = runner.invoke(
                app, ["dataset-qa", "check", "run-abc", "-o", "json"]
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == {
            "run_id": "run-abc",
            "cached_zip_path": "cache/run-abc.zip",
        }
        assert "Dataset QA Runs" not in result.stderr

    def test_sdk_error_emits_json_error_and_exits_1(self):
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.side_effect = HirundoError("boom")
            result = runner.invoke(app, ["dataset-qa", "run", "42", "-o", "json"])
        assert result.exit_code == 1
        assert json.loads(result.stdout) == {"error": "boom"}
        assert result.stderr == ""

    def test_http_error_emits_safe_json(self):
        sdk_error = requests.HTTPError("token=secret upstream detail")
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.side_effect = sdk_error
            result = runner.invoke(app, ["dataset-qa", "run", "42", "-o", "json"])
        assert result.exit_code == 1
        assert json.loads(result.stdout) == {"error": "HTTP request failed."}
        assert "secret" not in result.stdout

    def test_value_error_emits_json(self):
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.side_effect = ValueError("missing run ID")
            result = runner.invoke(app, ["dataset-qa", "run", "42", "-o", "json"])
        assert result.exit_code == 1
        assert json.loads(result.stdout) == {"error": "missing run ID"}

    def test_validation_error_emits_json_error(self):
        result = runner.invoke(app, ["dataset-qa", "check", "bad/id", "-o", "json"])
        assert result.exit_code == 1
        assert "Invalid run ID" in json.loads(result.stdout)["error"]

    @pytest.mark.parametrize(
        "arguments",
        [
            ["dataset-qa", "run", "not-an-int", "-o", "json"],
            ["dataset-qa", "run", "-o", "json"],
        ],
    )
    def test_typer_parse_error_emits_json(self, arguments):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 2
        assert "error" in json.loads(result.stdout)
        assert result.stderr == ""

    def test_missing_prompt_value_emits_json_without_prompting(self):
        result = runner.invoke(app, ["set-api-key", "-o", "json"])
        assert result.exit_code == 2
        assert json.loads(result.stdout) == {
            "error": "Missing required value for set-api-key in JSON mode."
        }
        assert "Please enter" not in result.stdout

    def test_missing_prompt_value_returns_exit_code_without_standalone_mode(
        self, capsys
    ):
        command = typer.main.get_command(app)

        exit_code = command.main(
            args=["set-api-key", "-o", "json"], standalone_mode=False
        )

        assert exit_code == 2
        assert json.loads(capsys.readouterr().out) == {
            "error": "Missing required value for set-api-key in JSON mode."
        }

    def test_malformed_json_help_uses_json_error_boundary(self, capsys):
        command = typer.main.get_command(app)
        with patch.object(
            TyperGroup,
            "main",
            side_effect=click.exceptions.UsageError("Malformed help arguments."),
        ):
            exit_code = command.main(
                args=["dataset-qa", "run", "-o", "json", "--help"],
                standalone_mode=False,
            )

        assert exit_code == 2
        assert json.loads(capsys.readouterr().out) == {
            "error": "Malformed help arguments."
        }

    def test_json_help_is_wrapped_in_json(self):
        result = runner.invoke(app, ["dataset-qa", "run", "-o", "json", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in json.loads(result.stdout)["help"]

    def test_json_help_precedes_prompt_validation(self):
        result = runner.invoke(app, ["set-api-key", "--help", "-o", "json"])
        assert result.exit_code == 0
        assert "Usage:" in json.loads(result.stdout)["help"]

    def test_root_positioned_output_option_is_not_treated_as_leaf_json(self):
        result = runner.invoke(app, ["-o", "json", "dataset-qa", "list"])
        assert result.exit_code == 2
        assert not result.stdout.lstrip().startswith("{")

    def test_json_mode_does_not_leak_into_later_text_invocation(self):
        json_result = runner.invoke(
            app, ["dataset-qa", "run", "not-an-int", "-o", "json"]
        )
        text_result = runner.invoke(app, ["dataset-qa", "check", "bad/id"])
        assert "error" in json.loads(json_result.stdout)
        assert "Invalid run ID" in text_result.stdout
        assert not text_result.stdout.lstrip().startswith("{")


class TestTextOutputUnaffected:
    def test_error_preserves_stdout_contract(self):
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.side_effect = HirundoError("boom")
            result = runner.invoke(app, ["dataset-qa", "run", "42"])
        assert result.exit_code == 1
        assert "boom" in result.stdout
        assert result.stderr == ""

    def test_check_result_preserves_plain_stdout_contract(self):
        check_results = MagicMock(cached_zip_path="cache/run-abc.zip")
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.check_run_by_id.return_value = check_results
            result = runner.invoke(app, ["check-run", "run-abc", "-t", "dataset-qa"])
        assert result.exit_code == 0
        assert result.stdout == "Run results saved to cache/run-abc.zip\n"

    def test_success_message_on_stdout(self):
        with patch("hirundo.dataset_qa.QADataset") as dataset_qa_mock:
            dataset_qa_mock.launch_qa_run.return_value = "run-abc"
            result = runner.invoke(app, ["dataset-qa", "run", "42", "--no-wait"])
        assert result.exit_code == 0
        assert "run-abc" in result.stdout
        # no JSON document in text mode
        assert "{" not in result.stdout


def test_table_cell_uses_compact_json():
    assert _cli_common._cell({"model": "gpt", "temperature": 0}) == (
        '{"model":"gpt","temperature":0}'
    )


def test_human_chatter_routes_to_stderr_in_json_mode():
    set_output_format(OutputFormat.json)
    assert _cli_common.is_json() is True
