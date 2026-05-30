import re
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

docs = "sphinx" in sys.modules
hirundo_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

console = Console()


def success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[green]{message}[/green]")


def info(message: str) -> None:
    """Print a plain informational message."""
    console.print(message)


def warn(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]{message}[/yellow]")


def error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[red]{message}[/red]")


def make_app(name: str, help_text: str) -> typer.Typer:
    return typer.Typer(
        name=name,
        no_args_is_help=True,
        rich_markup_mode="rich",
        epilog=hirundo_epilog,
        help=help_text,
    )


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        error(
            f"Invalid run ID '{run_id}'. Run IDs may only contain "
            "alphanumeric characters, hyphens, and underscores."
        )
        raise typer.Exit(code=1) from None
    return run_id


def validate_enum(value: str, enum_cls: type[Enum], label: str) -> Any:
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(member.value for member in enum_cls)
        error(f"Invalid {label} '{value}'. Valid options: {valid}.")
        raise typer.Exit(code=1) from None


def require_exactly_one(*options: tuple[str, Any]) -> None:
    """Exit with an error unless exactly one of the named options is set."""
    provided = [name for name, value in options if value is not None]
    if len(provided) != 1:
        names = " or ".join(name for name, _ in options)
        error(f"Exactly one of {names} must be provided.")
        raise typer.Exit(code=1) from None


def report_run_started(label: str, run_id: str) -> None:
    """Announce a freshly launched run in a consistent style."""
    success(f"{label} run started — Run ID: [bold]{run_id}[/bold]")


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


def check_run_and_print(run_id: str, check_fn: Callable[[str], Any]) -> None:
    _report_results(check_fn(validate_run_id(run_id)))


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
