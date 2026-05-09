from typing import Annotated

import typer
from rich.table import Table

from hirundo._cli_common import console, hirundo_epilog, make_app, validate_enum

eval_app = make_app("eval", "Launch and monitor LLM behavior evaluation runs.")


@eval_app.command("run", epilog=hirundo_epilog)
def eval_run(
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Evaluation preset. One of: BBQ_BIAS, BBQ_UNBIAS, UNQOVER_BIAS, HALU_EVAL, MED_HALLU, INJECTION_EVAL",
        ),
    ],
    model_id: Annotated[
        int | None,
        typer.Option("--model-id", help="ID of the LLM model to evaluate."),
    ] = None,
    source_run_id: Annotated[
        str | None,
        typer.Option("--source-run-id", help="ID of the unlearning run to evaluate."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Optional name for this evaluation run."),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait for the run to complete and stream progress."),
    ] = True,
):
    """
    Launch an LLM behavior evaluation run.

    Either --model-id or --source-run-id must be provided.
    """
    from hirundo.llm_behavior_eval import (
        EvalRunInfo,
        LlmBehaviorEval,
        ModelOrRun,
        PresetType,
    )

    if model_id is None and source_run_id is None:
        console.print("[red]Error: either --model-id or --source-run-id must be provided.[/red]")
        raise typer.Exit(code=1)
    if model_id is not None and source_run_id is not None:
        console.print("[red]Error: only one of --model-id or --source-run-id may be provided.[/red]")
        raise typer.Exit(code=1)

    preset_type = validate_enum(preset, PresetType, "preset")
    model_or_run = ModelOrRun.MODEL if model_id is not None else ModelOrRun.RUN
    run_info = EvalRunInfo(
        model_id=model_id,
        source_run_id=source_run_id,
        preset_type=preset_type,
        name=name,
    )

    run_id = LlmBehaviorEval.launch_eval_run(model_or_run, run_info)
    console.print(f"Eval run started. Run ID: [bold]{run_id}[/bold]")

    if wait:
        LlmBehaviorEval.check_run_by_id(run_id)
    else:
        console.print(
            "Use [bold]hirundo eval check[/bold] [italic]<run_id>[/italic] to monitor progress."
        )


@eval_app.command("list", epilog=hirundo_epilog)
def eval_list(
    archived: Annotated[
        bool,
        typer.Option("--archived/--no-archived", help="Include archived runs."),
    ] = False,
):
    """
    List LLM behavior evaluation runs.
    """
    from hirundo.llm_behavior_eval import LlmBehaviorEval

    runs = LlmBehaviorEval.list_runs(archived=archived)

    table = Table(title="Eval Runs:", expand=True)
    for col in ("Run ID", "Name", "Status", "Preset", "Created At"):
        table.add_column(col, overflow="fold")
    for run in runs:
        table.add_row(
            str(run.run_id),
            str(run.name),
            str(run.status),
            run.preset_type.value if run.preset_type else None,
            run.created_at.isoformat(),
        )
    console.print(table)


@eval_app.command("check", epilog=hirundo_epilog)
def eval_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of an LLM behavior evaluation run and stream progress.
    """
    from hirundo.llm_behavior_eval import LlmBehaviorEval

    LlmBehaviorEval.check_run_by_id(run_id)
