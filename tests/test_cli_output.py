import json
from unittest.mock import MagicMock, patch

import pytest
from hirundo import _cli_common
from hirundo._cli_common import OutputFormat, run_payload, set_output_format
from hirundo._hirundo_error import HirundoError
from hirundo.cli import app
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=False)


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
        with patch("hirundo.dataset_qa.QADataset") as qa:
            qa.launch_qa_run.return_value = "run-abc"
            result = runner.invoke(
                app, ["dataset-qa", "run", "42", "--no-wait", "-o", "json"]
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == {
            "run_id": "run-abc",
            "cached_zip_path": None,
        }

    def test_list_emits_json_array(self):
        run = MagicMock(name="ds", run_id="r1", status="COMPLETED", run_args=None)
        run.name = "ds"
        run.created_at.isoformat.return_value = "2026-05-31T00:00:00"
        with patch("hirundo.dataset_qa.QADataset") as qa:
            qa.list_runs.return_value = [run]
            result = runner.invoke(app, ["dataset-qa", "list", "-o", "json"])
        assert result.exit_code == 0
        assert json.loads(result.stdout) == [
            {
                "dataset_name": "ds",
                "run_id": "r1",
                "status": "COMPLETED",
                "created_at": "2026-05-31T00:00:00",
                "run_args": None,
            }
        ]

    def test_sdk_error_emits_json_error_and_exits_1(self):
        with patch("hirundo.dataset_qa.QADataset") as qa:
            qa.launch_qa_run.side_effect = HirundoError("boom")
            result = runner.invoke(app, ["dataset-qa", "run", "42", "-o", "json"])
        assert result.exit_code == 1
        assert json.loads(result.stdout) == {"error": "boom"}
        assert result.stderr == ""

    def test_validation_error_emits_json_error(self):
        result = runner.invoke(app, ["dataset-qa", "check", "bad/id", "-o", "json"])
        assert result.exit_code == 1
        assert "Invalid run ID" in json.loads(result.stdout)["error"]


class TestTextOutputUnaffected:
    def test_error_goes_to_stderr_not_stdout(self):
        with patch("hirundo.dataset_qa.QADataset") as qa:
            qa.launch_qa_run.side_effect = HirundoError("boom")
            result = runner.invoke(app, ["dataset-qa", "run", "42"])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "boom" in result.stderr

    def test_success_message_on_stdout(self):
        with patch("hirundo.dataset_qa.QADataset") as qa:
            qa.launch_qa_run.return_value = "run-abc"
            result = runner.invoke(app, ["dataset-qa", "run", "42", "--no-wait"])
        assert result.exit_code == 0
        assert "run-abc" in result.stdout
        # no JSON document in text mode
        assert "{" not in result.stdout


def test_human_chatter_routes_to_stderr_in_json_mode():
    set_output_format(OutputFormat.json)
    assert _cli_common.is_json() is True
