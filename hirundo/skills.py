"""
CLI commands for installing Hirundo coding assistant skills.

Usage:
    hirundo skills install claude-code   # install into .claude/commands/ (project-local)
    hirundo skills install cursor        # install into .cursor/rules/ (project-local)
    hirundo skills install opencode      # install into .opencode/rules/ (project-local)
    hirundo skills install codex         # append section to AGENTS.md (project-local)
    hirundo skills install claude-code --global  # install into ~/.claude/commands/
    hirundo skills install codex --global        # append to ~/.codex/AGENTS.md
"""

import importlib.resources
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

docs = "sphinx" in sys.modules
skills_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

console = Console()

skills_app = typer.Typer(
    name="skills",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=skills_epilog,
    help="Install Hirundo assistant skills for your coding assistant.",
)

_SKILL_NAME = "hirundo-assistant"
_SKILL_FILE = f"{_SKILL_NAME}.md"

# Sentinel written into AGENTS.md so re-runs can detect and replace the section.
_CODEX_SECTION_START = "<!-- hirundo-assistant:start -->"
_CODEX_SECTION_END = "<!-- hirundo-assistant:end -->"

_TOOLS: dict[str, dict] = {
    "claude-code": {
        "local_dir": Path(".claude") / "commands",
        "global_dir": Path.home() / ".claude" / "commands",
        "extension": ".md",
        "label": "Claude Code",
    },
    "cursor": {
        "local_dir": Path(".cursor") / "rules",
        "global_dir": Path.home() / ".cursor" / "rules",
        "extension": ".mdc",
        "label": "Cursor",
    },
    "opencode": {
        "local_dir": Path(".opencode") / "rules",
        "global_dir": Path.home() / ".opencode" / "rules",
        "extension": ".md",
        "label": "OpenCode",
    },
    "codex": {
        # Codex reads AGENTS.md files; no dedicated skill directory.
        "local_agents_md": Path("AGENTS.md"),
        "global_agents_md": Path.home() / ".codex" / "AGENTS.md",
        "label": "Codex",
    },
}


def _read_skill_content() -> str:
    pkg = importlib.resources.files("hirundo") / "skills_data" / _SKILL_FILE
    return pkg.read_text(encoding="utf-8")


def _cursor_frontmatter(content: str) -> str:
    """Wrap skill content with Cursor-compatible frontmatter."""
    frontmatter = (
        "---\n"
        "description: Interactive guide for the Hirundo Python SDK\n"
        "alwaysApply: false\n"
        "---\n\n"
    )
    return frontmatter + content


def _install_codex(agents_md: Path, content: str) -> None:
    """
    Upsert a fenced Hirundo section inside an AGENTS.md file.

    If a previous install section is present it is replaced in-place;
    otherwise the section is appended.
    """
    section = f"{_CODEX_SECTION_START}\n{content}\n{_CODEX_SECTION_END}"

    if agents_md.exists():
        existing = agents_md.read_text(encoding="utf-8")
        if _CODEX_SECTION_START in existing:
            # Replace the existing section between the sentinels.
            import re

            updated = re.sub(
                rf"{re.escape(_CODEX_SECTION_START)}.*?{re.escape(_CODEX_SECTION_END)}",
                section,
                existing,
                flags=re.DOTALL,
            )
            agents_md.write_text(updated, encoding="utf-8")
            return
        # Append after a blank line separator.
        separator = "\n\n" if existing.rstrip() else ""
        agents_md.write_text(existing.rstrip() + separator + "\n" + section + "\n", encoding="utf-8")
    else:
        agents_md.parent.mkdir(parents=True, exist_ok=True)
        agents_md.write_text(section + "\n", encoding="utf-8")


@skills_app.command("install", epilog=skills_epilog)
def install(
    tool: Annotated[
        str,
        typer.Argument(
            help=f"Coding assistant to install for. One of: {', '.join(_TOOLS)}",
            metavar="TOOL",
        ),
    ],
    global_install: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Install globally (user home directory) instead of project-local.",
        ),
    ] = False,
):
    """
    Install the Hirundo assistant skill for your coding assistant.

    After installing, the skill is available as /hirundo-assistant in your editor
    (claude-code, cursor, opencode) or as part of AGENTS.md context (codex).

    Supported tools: claude-code, cursor, opencode, codex
    """
    tool = tool.lower()
    if tool not in _TOOLS:
        console.print(
            f"[red]Unknown tool '{tool}'. Choose from: {', '.join(_TOOLS)}[/red]"
        )
        raise typer.Exit(code=1)

    cfg = _TOOLS[tool]
    content = _read_skill_content()
    scope = "globally" if global_install else "project-locally"

    if tool == "codex":
        agents_md: Path = cfg["global_agents_md"] if global_install else cfg["local_agents_md"]
        _install_codex(agents_md, content)
        console.print(
            f"[green]✓[/green] Hirundo assistant skill installed {scope} for {cfg['label']}."
        )
        console.print(f"  Location: [bold]{agents_md}[/bold]")
        if not global_install:
            console.print(
                "  Tip: commit AGENTS.md so your whole team has access to the skill."
            )
        return

    dest_dir: Path = cfg["global_dir"] if global_install else cfg["local_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext: str = cfg["extension"]
    dest_file = dest_dir / f"{_SKILL_NAME}{ext}"

    if tool == "cursor":
        content = _cursor_frontmatter(content)

    dest_file.write_text(content, encoding="utf-8")

    console.print(
        f"[green]✓[/green] Hirundo assistant skill installed {scope} for {cfg['label']}."
    )
    console.print(f"  Location: [bold]{dest_file}[/bold]")
    if not global_install:
        console.print(
            "  Tip: commit this file so your whole team has access to the skill."
        )
    console.print(
        f"\n  Invoke it in your editor with: [bold]/hirundo-assistant[/bold]"
    )


@skills_app.command("list", epilog=skills_epilog)
def list_skills():
    """
    List available Hirundo skills.
    """
    console.print("[bold]Available Hirundo skills:[/bold]\n")
    console.print(f"  [cyan]{_SKILL_NAME}[/cyan]")
    console.print(
        "    Interactive guide for LLM behavior evals, Dataset QA, and LLM Unlearning.\n"
    )
    console.print(
        f"Install with: [bold]hirundo skills install <tool>[/bold]"
    )
    console.print(f"Supported tools: {', '.join(_TOOLS)}")
