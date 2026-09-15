import json
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from enum import Enum
from io import StringIO
from typing import Annotated, Any, NoReturn, TypeAlias, TypeVar

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from typer import _click as click
from typer.core import TyperGroup

from hirundo._hirundo_error import HirundoError
from hirundo._http import requests

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_RunResult = TypeVar("_RunResult")

docs = "sphinx" in sys.modules
hirundo_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

# Human-facing output goes to stdout in text mode; in JSON mode stdout is
# reserved for the machine-readable document, so all chatter moves to stderr.
console = Console()
err_console = Console(stderr=True)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


_output_format = OutputFormat.text


def set_output_format(output_format: OutputFormat) -> None:
    global _output_format
    _output_format = output_format


def is_json() -> bool:
    return _output_format is OutputFormat.json


OutputOption: TypeAlias = Annotated[
    OutputFormat,
    typer.Option(
        "--output",
        "-o",
        help="Output format. Use 'json' for machine-readable output.",
    ),
]


def _human_console() -> Console:
    """Where human-facing chatter goes: stderr in JSON mode, else stdout."""
    return err_console if is_json() else console


def emit_json(data: Any) -> None:
    """Write a single machine-readable JSON document to stdout."""
    json.dump(data, sys.stdout, default=str, indent=2)
    sys.stdout.write("\n")


def emit_if_json(data: Any) -> None:
    """Emit ``data`` as JSON only in JSON mode (no-op in text mode)."""
    if is_json():
        emit_json(data)


def success(message: str) -> None:
    """Print a success message in green."""
    _human_console().print(f"[green]{message}[/green]")


def info(message: str) -> None:
    """Print a plain informational message."""
    _human_console().print(message)


def warn(message: str) -> None:
    """Print a warning message in yellow."""
    _human_console().print(f"[yellow]{message}[/yellow]")


def error(message: str) -> None:
    """Print an error message on the human-facing output stream."""
    _human_console().print(f"[red]{message}[/red]")


def _emit_error(message: str) -> None:
    """Report an error as JSON on stdout (JSON mode) or red on stderr (text)."""
    if is_json():
        emit_json({"error": message})
    else:
        error(message)


def fail(message: str) -> NoReturn:
    """Report an error and exit with code 1."""
    _emit_error(message)
    raise typer.Exit(code=1)


ArchivedOption: TypeAlias = Annotated[
    bool,
    typer.Option("--archived/--no-archived", help="Include archived runs."),
]

WaitOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--wait/--no-wait", help="Wait for the run to complete and stream progress."
    ),
]


class HirundoCliGroup(TyperGroup):
    """Typer group that turns SDK errors into clean CLI output + exit code 1."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        """Run the CLI with JSON-aware handling for parsing failures.

        Args:
            args: Command-line arguments, or ``sys.argv[1:]`` when omitted.
            prog_name: Program name shown in usage and error messages.
            complete_var: Environment variable used for shell completion.
            standalone_mode: Whether exits should raise ``SystemExit``.
            windows_expand_args: Whether to expand glob and user-path arguments on
                Windows.
            **extra: Additional context settings forwarded to Typer.

        Returns:
            The command callback result when execution completes without exiting.
        """
        arguments = list(sys.argv[1:] if args is None else args)
        output_format = _detect_output_format(arguments)
        set_output_format(output_format)
        if output_format is not OutputFormat.json:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )

        if self._handle_json_preflight(
            arguments,
            args,
            prog_name,
            complete_var,
            standalone_mode,
            windows_expand_args,
            extra,
        ):
            return None

        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
            if isinstance(result, int) and result != 0:
                if standalone_mode:
                    raise SystemExit(result)
                raise click.exceptions.Exit(result)
            return result
        except click.ClickException as error_value:
            _emit_error(error_value.format_message())
            if standalone_mode:
                raise SystemExit(error_value.exit_code) from error_value
            raise click.exceptions.Exit(error_value.exit_code) from error_value
        except click.exceptions.Exit as exit_value:
            if standalone_mode:
                raise SystemExit(exit_value.exit_code) from exit_value
            raise

    def _handle_json_preflight(
        self,
        arguments: Sequence[str],
        args: Sequence[str] | None,
        prog_name: str | None,
        complete_var: str | None,
        standalone_mode: bool,
        windows_expand_args: bool,
        extra: dict[str, Any],
    ) -> bool:
        if any(argument in {"--help", "-h"} for argument in arguments):
            help_buffer = StringIO()
            try:
                with redirect_stdout(help_buffer):
                    super().main(
                        args=args,
                        prog_name=prog_name,
                        complete_var=complete_var,
                        standalone_mode=False,
                        windows_expand_args=windows_expand_args,
                        **extra,
                    )
            except click.exceptions.Exit as exit_value:
                if exit_value.exit_code != 0:
                    raise
            emit_json({"help": help_buffer.getvalue()})
            return True
        prompted_error = _missing_json_prompt_value(arguments)
        if prompted_error is None:
            return False
        _emit_error(prompted_error)
        if standalone_mode:
            raise SystemExit(2)
        raise click.exceptions.Exit(2)

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except requests.HTTPError as error_value:
            response = error_value.response
            status_suffix = (
                f" with status {response.status_code}" if response is not None else ""
            )
            _emit_error(f"HTTP request failed{status_suffix}.")
            raise typer.Exit(code=1) from error_value
        except (HirundoError, ValueError) as error_value:
            _emit_error(str(error_value))
            raise typer.Exit(code=1) from error_value
        except click.ClickException as error_value:
            if not is_json():
                raise
            _emit_error(error_value.format_message())
            raise click.exceptions.Exit(error_value.exit_code) from error_value


def _detect_output_format(arguments: Sequence[str]) -> OutputFormat:
    """Detect JSON mode before Click validates subcommand arguments."""
    nested_groups = {"dataset-qa", "eval", "unlearning"}
    leaf_index = 1 if arguments and arguments[0] in nested_groups else 0
    leaf_arguments = arguments[leaf_index + 1 :]
    for argument_index, argument in enumerate(leaf_arguments):
        if argument == "--output=json" or argument == "-ojson":
            return OutputFormat.json
        if argument in {"--output", "-o"} and argument_index + 1 < len(leaf_arguments):
            if leaf_arguments[argument_index + 1] == OutputFormat.json.value:
                return OutputFormat.json
    return OutputFormat.text


def _missing_json_prompt_value(arguments: Sequence[str]) -> str | None:
    """Return an error when JSON mode would otherwise start an interactive prompt.

    Args:
        arguments: Command-line arguments being prepared for Click validation.

    Returns:
        An error message when a prompted value is missing, otherwise ``None``.
    """
    command_value_counts = {"set-api-key": 1, "change-remote": 1, "setup": 2}
    if not arguments or arguments[0] not in command_value_counts:
        return None
    positional_values: list[str] = []
    skip_next = False
    for argument in arguments[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"--output", "-o"}:
            skip_next = True
            continue
        if argument.startswith("--output=") or argument.startswith("-o"):
            continue
        if not argument.startswith("-"):
            positional_values.append(argument)
    required_count = command_value_counts[arguments[0]]
    if len(positional_values) < required_count:
        return f"Missing required value for {arguments[0]} in JSON mode."
    return None


def make_app(name: str, help_text: str) -> typer.Typer:
    return typer.Typer(
        name=name,
        cls=HirundoCliGroup,
        no_args_is_help=True,
        rich_markup_mode="rich",
        epilog=hirundo_epilog,
        help=help_text,
    )


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        fail(
            f"Invalid run ID '{run_id}'. Run IDs may only contain "
            "alphanumeric characters, hyphens, and underscores."
        )
    return run_id


def validate_enum(value: str, enum_cls: type[Enum], label: str) -> Any:
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(member.value for member in enum_cls)
        fail(f"Invalid {label} '{value}'. Valid options: {valid}.")


def require_exactly_one(*options: tuple[str, Any]) -> None:
    """Exit with an error unless exactly one of the named options is set."""
    provided = [name for name, value in options if value is not None]
    if len(provided) != 1:
        names = " or ".join(option_name for option_name, _option_value in options)
        fail(f"Exactly one of {names} must be provided.")


def report_run_started(label: str, run_id: str) -> None:
    """Announce a freshly launched run in a consistent style."""
    success(f"{label} run started — Run ID: [bold]{run_id}[/bold]")


def _cached_zip_path(results: Any) -> str | None:
    path = getattr(results, "cached_zip_path", None)
    return str(path) if path is not None else None


def run_payload(run_id: str, results: Any = None) -> dict[str, Any]:
    """Machine-readable payload describing a run and (optionally) its results."""
    return {"run_id": run_id, "cached_zip_path": _cached_zip_path(results)}


def _report_results(results: _RunResult) -> _RunResult:
    cached_zip_path = _cached_zip_path(results)
    if cached_zip_path is not None:
        _human_console().print(f"Run results saved to {cached_zip_path}")
    return results


def wait_or_notify(
    run_id: str,
    check_fn: Callable[[str], _RunResult],
    cmd_name: str,
    wait: bool,
) -> _RunResult | None:
    if not wait:
        info(
            f"Use [bold]hirundo {cmd_name} check[/bold] [italic]<run_id>[/italic] "
            "to monitor progress."
        )
        return None
    return _report_results(check_fn(run_id))


def wait_and_emit_run(
    run_id: str,
    check_function: Callable[[str], _RunResult],
    command_name: str,
    wait: bool,
) -> _RunResult | None:
    """Wait for a launched run when requested, then emit its JSON payload.

    Args:
        run_id: Identifier returned by the launch operation.
        check_function: SDK function that waits for and returns the run result.
        command_name: CLI command group shown in the text-mode progress hint.
        wait: Whether to wait for completion before returning.

    Returns:
        The SDK result when waiting, otherwise ``None``.
    """
    results = wait_or_notify(run_id, check_function, command_name, wait)
    emit_if_json(run_payload(run_id, results))
    return results


def check_run_and_print(
    run_id: str,
    check_fn: Callable[[str], _RunResult],
    *,
    validate: bool = True,
) -> _RunResult:
    checked_run_id = validate_run_id(run_id) if validate else run_id
    return _report_results(check_fn(checked_run_id))


def check_and_emit_run(
    run_id: str,
    check_function: Callable[[str], _RunResult],
    *,
    validate: bool = True,
) -> _RunResult:
    """Check a run and emit the shared JSON payload when requested.

    Args:
        run_id: Run identifier supplied by the user.
        check_function: SDK function that checks and returns the run result.
        validate: Whether to apply the shared run-ID validator before checking.

    Returns:
        The result returned by ``check_function``.
    """
    results = check_run_and_print(run_id, check_function, validate=validate)
    emit_if_json(run_payload(run_id, results))
    return results


def _cell(value: Any) -> str | None:
    """Render a value for a table cell (objects become compact JSON)."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def emit_rows(
    title: str, columns: list[tuple[str, str]], items: list[dict[str, Any]]
) -> None:
    """List output: a JSON array in JSON mode, else a Rich table.

    ``columns`` maps each table header to the item key it renders.
    """
    if is_json():
        emit_json(items)
        return

    table = Table(
        title=title,
        box=box.SIMPLE,
        show_lines=False,
        show_edge=True,
        header_style="bold",
    )
    for column_header, _column_key in columns:
        table.add_column(column_header, overflow="fold")
    for row_item in items:
        table.add_row(
            *(_cell(row_item[column_key]) for _column_header, column_key in columns)
        )
    console.print(table)
