import re
import sys

import typer
from rich.console import Console

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
    if not _RUN_ID_RE.match(run_id):
        console.print(
            f"[red]Invalid run ID '{run_id}'. "
            "Run IDs may only contain alphanumeric characters, hyphens, and underscores.[/red]"
        )
        raise typer.Exit(code=1) from None
    return run_id


def validate_enum(value: str, enum_cls, label: str):
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(e.value for e in enum_cls)
        console.print(f"[red]Invalid {label} '{value}'. Valid options: {valid}[/red]")
        raise typer.Exit(code=1) from None
