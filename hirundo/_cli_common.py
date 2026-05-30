import json
import re
import sys
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, NoReturn, TypeAlias

import click
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from hirundo._hirundo_error import HirundoError

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

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


def set_output_format(fmt: OutputFormat) -> None:
    global _output_format
    _output_format = fmt


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


def emit(data: Any, render_text: Callable[[], None]) -> None:
    """Emit ``data`` as JSON in JSON mode, otherwise call ``render_text``."""
    if is_json():
        emit_json(data)
    else:
        render_text()


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
    """Print an error message in red (always to stderr)."""
    err_console.print(f"[red]{message}[/red]")


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

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except HirundoError as exc:
            _emit_error(str(exc))
            raise typer.Exit(code=1) from exc


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
        names = " or ".join(name for name, _ in options)
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


def _report_results(results: Any) -> Any:
    if results is not None and hasattr(results, "cached_zip_path"):
        success(f"Run results saved to [bold]{results.cached_zip_path}[/bold]")
    return results


def wait_or_notify(
    run_id: str, check_fn: Callable[[str], Any], cmd_name: str, wait: bool
) -> Any:
    if not wait:
        info(
            f"Use [bold]hirundo {cmd_name} check[/bold] [italic]<run_id>[/italic] "
            "to monitor progress."
        )
        return None
    return _report_results(check_fn(run_id))


def check_run_and_print(run_id: str, check_fn: Callable[[str], Any]) -> Any:
    return _report_results(check_fn(validate_run_id(run_id)))


def print_runs_table(
    title: str,
    columns: tuple[str, ...],
    rows: list[tuple[str | None, ...]],
) -> None:
    table = Table(
        title=title,
        box=box.SIMPLE,
        show_lines=False,
        show_edge=True,
        header_style="bold",
    )
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(*row)
    console.print(table)
