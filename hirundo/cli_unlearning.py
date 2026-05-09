from typing import Annotated

import typer
from rich.table import Table

from hirundo._cli_common import (
    console,
    hirundo_epilog,
    make_app,
    validate_enum,
    validate_run_id,
)

unlearning_app = make_app("unlearning", "Launch and monitor LLM unlearning runs.")


@unlearning_app.command("run", epilog=hirundo_epilog)
def unlearning_run(
    model_id: Annotated[int, typer.Argument(help="ID of the LLM model to unlearn.")],
    bias_type: Annotated[
        str | None,
        typer.Option(
            "--bias-type",
            help="Bias type for unlearning. One of: ALL, RACE, NATIONALITY, GENDER, PHYSICAL_APPEARANCE, RELIGION, AGE",
        ),
    ] = None,
    hallucination_type: Annotated[
        str | None,
        typer.Option(
            "--hallucination-type",
            help="Hallucination type for unlearning. One of: GENERAL, MEDICAL, LEGAL, DEFENSE",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Optional name for this unlearning run."),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait", help="Wait for the run to complete and stream progress."
        ),
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
        target_behavior = BiasBehavior(
            bias_type=validate_enum(bias_type, BBQBiasType, "bias type")
        )
    elif hallucination_type is not None:
        target_behavior = HallucinationBehavior(
            hallucination_type=validate_enum(
                hallucination_type, HallucinationType, "hallucination type"
            )
        )
    else:
        raise typer.Exit(code=1) from None

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


@unlearning_app.command("list", epilog=hirundo_epilog)
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


@unlearning_app.command("check", epilog=hirundo_epilog)
def unlearning_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of an LLM unlearning run and stream progress.
    """
    from hirundo.unlearning_llm import LlmUnlearningRun

    LlmUnlearningRun.check_run_by_id(validate_run_id(run_id))
