import sys

import typer
from rich.console import Console

docs = "sphinx" in sys.modules
hirundo_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

console = Console()


def make_app(name: str, help: str) -> typer.Typer:
    return typer.Typer(
        name=name,
        no_args_is_help=True,
        rich_markup_mode="rich",
        epilog=hirundo_epilog,
        help=help,
    )


def validate_enum(value: str, enum_cls, label: str):
    try:
        return enum_cls(value.upper())
    except ValueError:
        valid = ", ".join(e.value for e in enum_cls)
        console.print(f"[red]Invalid {label} '{value}'. Valid options: {valid}[/red]")
        raise typer.Exit(code=1)
