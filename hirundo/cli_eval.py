from typing import Annotated

import typer

from hirundo._cli_common import (
    ArchivedOption,
    OutputFormat,
    OutputOption,
    WaitOption,
    check_and_emit_run,
    emit_rows,
    hirundo_epilog,
    make_app,
    report_run_started,
    require_exactly_one,
    set_output_format,
    validate_enum,
    validate_run_id,
    wait_and_emit_run,
)

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
    wait: WaitOption = True,
    output: OutputOption = OutputFormat.text,
):
    """
    Launch an LLM behavior evaluation run.

    Either --model-id or --source-run-id must be provided.
    """
    set_output_format(output)
    from hirundo.llm_behavior_eval import (
        EvalRunInfo,
        LlmBehaviorEval,
        ModelOrRun,
        PresetType,
    )

    require_exactly_one(("--model-id", model_id), ("--source-run-id", source_run_id))

    if source_run_id is not None:
        source_run_id = validate_run_id(source_run_id)

    preset_type = validate_enum(preset, PresetType, "preset")
    model_or_run = ModelOrRun.MODEL if model_id is not None else ModelOrRun.RUN
    run_info = EvalRunInfo(
        model_id=model_id,
        source_run_id=source_run_id,
        preset_type=preset_type,
        name=name,
    )

    run_id = LlmBehaviorEval.launch_eval_run(model_or_run, run_info)
    report_run_started("Eval", run_id)

    wait_and_emit_run(run_id, LlmBehaviorEval.check_run_by_id, "eval", wait)


@eval_app.command("list", epilog=hirundo_epilog)
def eval_list(
    archived: ArchivedOption = False,
    output: OutputOption = OutputFormat.text,
):
    """
    List LLM behavior evaluation runs.
    """
    set_output_format(output)
    from hirundo.llm_behavior_eval import LlmBehaviorEval

    run_records = LlmBehaviorEval.list_runs(archived=archived)
    items = [
        {
            "run_id": str(run_record.run_id),
            "name": str(run_record.name),
            "status": str(run_record.status),
            "preset": (
                run_record.preset_type.value if run_record.preset_type else None
            ),
            "created_at": run_record.created_at.isoformat(),
        }
        for run_record in run_records
    ]
    emit_rows(
        "Eval Runs:",
        [
            ("Run ID", "run_id"),
            ("Name", "name"),
            ("Status", "status"),
            ("Preset", "preset"),
            ("Created At", "created_at"),
        ],
        items,
    )


@eval_app.command("check", epilog=hirundo_epilog)
def eval_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
    output: OutputOption = OutputFormat.text,
):
    """
    Check the status of an LLM behavior evaluation run and stream progress.
    """
    set_output_format(output)
    from hirundo.llm_behavior_eval import LlmBehaviorEval

    check_and_emit_run(run_id, LlmBehaviorEval.check_run_by_id)
