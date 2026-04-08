"""
CLI sub-app for LLM unlearning commands.

Commands:
    hirundo unlearning run    - Launch an LLM unlearning run
    hirundo unlearning list   - List LLM unlearning runs
    hirundo unlearning check  - Check the status of an LLM unlearning run
"""

import sys
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

docs = "sphinx" in sys.modules
unlearning_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

console = Console()

unlearning_app = typer.Typer(
    name="unlearning",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=unlearning_epilog,
    help="Launch and monitor LLM unlearning runs.",
)


@unlearning_app.command("run", epilog=unlearning_epilog)
def unlearning_run(
    model_id: Annotated[int, typer.Argument(help="ID of the LLM model to unlearn.")],
    bias_type: Annotated[
        Optional[str],
        typer.Option(
            "--bias-type",
            help="Bias type for unlearning. One of: ALL, RACE, NATIONALITY, GENDER, PHYSICAL_APPEARANCE, RELIGION, AGE",
        ),
    ] = None,
    hallucination_type: Annotated[
        Optional[str],
        typer.Option(
            "--hallucination-type",
            help="Hallucination type for unlearning. One of: GENERAL, MEDICAL, LEGAL, DEFENSE",
        ),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", help="Optional name for this unlearning run."),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait for the run to complete and stream progress."),
    ] = True,
):
    """
    Launch an LLM unlearning run.

    Exactly one of --bias-type or --hallucination-type must be provided.
    """
    from hirundo.llm_bias_type import BBQBiasType
    from hirundo.unlearning_llm import (
        BiasBehavior,
        DefaultUtility,
        HallucinationBehavior,
        HallucinationType,
        LlmRunInfo,
        LlmUnlearningRun,
    )

    if bias_type is None and hallucination_type is None:
        console.print(
            "[red]Error: either --bias-type or --hallucination-type must be provided.[/red]"
        )
        raise typer.Exit(code=1)
    if bias_type is not None and hallucination_type is not None:
        console.print(
            "[red]Error: only one of --bias-type or --hallucination-type may be provided.[/red]"
        )
        raise typer.Exit(code=1)

    if bias_type is not None:
        try:
            bias_type_enum = BBQBiasType(bias_type.upper())
        except ValueError:
            valid = ", ".join(b.value for b in BBQBiasType)
            console.print(
                f"[red]Invalid bias type '{bias_type}'. Valid options: {valid}[/red]"
            )
            raise typer.Exit(code=1)
        target_behavior = BiasBehavior(bias_type=bias_type_enum)
    else:
        try:
            hallucination_type_enum = HallucinationType(hallucination_type.upper())
        except ValueError:
            valid = ", ".join(h.value for h in HallucinationType)
            console.print(
                f"[red]Invalid hallucination type '{hallucination_type}'. Valid options: {valid}[/red]"
            )
            raise typer.Exit(code=1)
        target_behavior = HallucinationBehavior(hallucination_type=hallucination_type_enum)

    run_info = LlmRunInfo(
        name=name,
        target_behaviors=[target_behavior],
        target_utilities=[DefaultUtility()],
    )

    run_id = LlmUnlearningRun.launch(model_id, run_info)
    console.print(f"Unlearning run started. Run ID: [bold]{run_id}[/bold]")

    if wait:
        LlmUnlearningRun.check_run_by_id(run_id)
    else:
        console.print(
            "Use [bold]hirundo unlearning check[/bold] [italic]<run_id>[/italic] to monitor progress."
        )


@unlearning_app.command("list", epilog=unlearning_epilog)
def unlearning_list(
    archived: Annotated[
        bool,
        typer.Option("--archived/--no-archived", help="Include archived runs."),
    ] = False,
):
    """
    List LLM unlearning runs.
    """
    from hirundo.unlearning_llm import LlmUnlearningRun

    runs = LlmUnlearningRun.list(archived=archived)

    table = Table(title="Unlearning Runs:", expand=True)
    for col in ("Name", "Run ID", "Status", "Created At"):
        table.add_column(col, overflow="fold")
    for run in runs:
        table.add_row(
            str(run.name),
            str(run.run_id),
            str(run.status),
            run.created_at.isoformat(),
        )
    console.print(table)


@unlearning_app.command("check", epilog=unlearning_epilog)
def unlearning_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of an LLM unlearning run and stream progress.
    """
    from hirundo.unlearning_llm import LlmUnlearningRun

    LlmUnlearningRun.check_run_by_id(run_id)
