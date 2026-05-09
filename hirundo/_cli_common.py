import re
import sys
from collections.abc import Callable
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


def validate_enum(value: str, enum_cls: type, label: str) -> Any:
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(e.value for e in enum_cls)
        console.print(f"[red]Invalid {label} '{value}'. Valid options: {valid}[/red]")
        raise typer.Exit(code=1) from None


def wait_or_notify(
    run_id: str, check_fn: Callable[[str], Any], cmd_name: str, wait: bool
) -> Any:
    if wait:
        return check_fn(run_id)
    console.print(
        f"Use [bold]hirundo {cmd_name} check[/bold] [italic]<run_id>[/italic] to monitor progress."
    )
    return None


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
