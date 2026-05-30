import re
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any

import typer
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
        console.print(
            f"[red]Invalid run ID '{run_id}'. "
            "Run IDs may only contain alphanumeric characters, hyphens, and underscores.[/red]"
        )
        raise typer.Exit(code=1) from None
    return run_id


def validate_enum(value: str, enum_cls: type[Enum], label: str) -> Any:
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(member.value for member in enum_cls)
        console.print(f"[red]Invalid {label} '{value}'. Valid options: {valid}[/red]")
        raise typer.Exit(code=1) from None


def require_exactly_one(*options: tuple[str, Any]) -> None:
    """Exit with an error unless exactly one of the named options is set."""
    provided = [name for name, value in options if value is not None]
    if len(provided) != 1:
        names = " or ".join(name for name, _ in options)
        console.print(f"[red]Error: exactly one of {names} must be provided.[/red]")
        raise typer.Exit(code=1) from None


def _report_results(results: Any) -> Any:
    if results is not None:
        console.print(f"Run results saved to {results.cached_zip_path}")
    return results


def wait_or_notify(
    run_id: str, check_fn: Callable[[str], Any], cmd_name: str, wait: bool
) -> Any:
    if not wait:
        console.print(
            f"Use [bold]hirundo {cmd_name} check[/bold] [italic]<run_id>[/italic] to monitor progress."
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
    table = Table(title=title, expand=True)
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(*row)
    console.print(table)
