from unittest.mock import MagicMock, patch

import hirundo._cli_common as cli_common  # noqa: E402
import pytest
import typer
from hirundo._cli_common import (
    require_exactly_one,
    validate_run_id,
    wait_or_notify,
)


class TestRequireExactlyOne:
    def test_exactly_one_set_is_ok(self):
        require_exactly_one(("--a", "x"), ("--b", None))

    @pytest.mark.parametrize(
        "options",
        [
            ((("--a", None), ("--b", None))),
            ((("--a", "x"), ("--b", "y"))),
        ],
    )
    def test_zero_or_many_exits(self, options):
        with pytest.raises(typer.Exit) as exc:
            require_exactly_one(*options)
        assert exc.value.exit_code == 1

    def test_error_message_lists_option_names(self):
        with (
            patch.object(cli_common.console, "print") as mock_print,
            pytest.raises(typer.Exit),
        ):
            require_exactly_one(("--a", None), ("--b", None))
        output = mock_print.call_args[0][0]
        assert "--a or --b" in output


class TestValidateRunId:
    def test_valid_id_returned_unchanged(self):
        assert validate_run_id("abc-123_XYZ") == "abc-123_XYZ"

    @pytest.mark.parametrize(
        "bad_id", ["run/id", "run\\id", "run id", "run\nid", "run.id", ""]
    )
    def test_invalid_id_exits(self, bad_id):
        with pytest.raises(typer.Exit) as exc:
            validate_run_id(bad_id)
        assert exc.value.exit_code == 1

    def test_invalid_id_prints_message(self, bad_id="bad id"):
        with (
            patch.object(cli_common.console, "print") as mock_print,
            pytest.raises(typer.Exit),
        ):
            validate_run_id(bad_id)
        output = mock_print.call_args[0][0]
        assert "bad id" in output
        assert "may only contain" in output


class TestWaitOrNotify:
    def test_wait_true_calls_check_fn_and_returns_result(self):
        result = MagicMock(cached_zip_path=None)
        result.__bool__ = lambda self: True
        check_fn = MagicMock(return_value=result)
        assert wait_or_notify("run-1", check_fn, "dataset-qa", wait=True) is result
        check_fn.assert_called_once_with("run-1")

    def test_wait_true_prints_results_path_when_present(self):
        result = MagicMock(cached_zip_path="cache/run-1.zip")
        check_fn = MagicMock(return_value=result)
        with patch.object(cli_common.console, "print") as mock_print:
            wait_or_notify("run-1", check_fn, "dataset-qa", wait=True)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "cache/run-1.zip" in printed

    def test_wait_true_no_print_when_results_none(self):
        check_fn = MagicMock(return_value=None)
        with patch.object(cli_common.console, "print") as mock_print:
            wait_or_notify("run-1", check_fn, "dataset-qa", wait=True)
        mock_print.assert_not_called()

    def test_wait_false_returns_none_without_calling_check_fn(self):
        check_fn = MagicMock()
        assert wait_or_notify("run-1", check_fn, "dataset-qa", wait=False) is None
        check_fn.assert_not_called()

    def test_wait_false_prints_check_hint(self):
        with patch.object(cli_common.console, "print") as mock_print:
            wait_or_notify("run-1", MagicMock(), "dataset-qa", wait=False)
        output = mock_print.call_args[0][0]
        assert "dataset-qa check" in output
