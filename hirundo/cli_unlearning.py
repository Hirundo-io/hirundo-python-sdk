from typing import Annotated

import typer

from hirundo._cli_common import (
    ArchivedOption,
    OutputFormat,
    OutputOption,
    WaitOption,
    check_run_and_print,
    emit_if_json,
    emit_rows,
    hirundo_epilog,
    make_app,
    report_run_started,
    require_exactly_one,
    run_payload,
    set_output_format,
    validate_enum,
    wait_or_notify,
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
    wait: WaitOption = True,
    output: OutputOption = OutputFormat.text,
):
    """
    Launch an LLM unlearning run.

    Exactly one of --bias-type or --hallucination-type must be provided.
    """
    set_output_format(output)
    from hirundo.llm_bias_type import BBQBiasType
    from hirundo.unlearning_llm import (
        BiasBehavior,
        HallucinationBehavior,
        HallucinationType,
        LlmRunInfo,
        LlmUnlearningRun,
    )

    require_exactly_one(
        ("--bias-type", bias_type), ("--hallucination-type", hallucination_type)
    )

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
    else:  # unreachable: require_exactly_one guarantees one is set
        raise typer.Exit(code=1) from None

    run_info = LlmRunInfo(
        name=name,
        target_behaviors=[target_behavior],
    )

    run_id = LlmUnlearningRun.launch(model_id, run_info)
    report_run_started("Unlearning", run_id)

    results = wait_or_notify(
        run_id, LlmUnlearningRun.check_run_by_id, "unlearning", wait
    )
    emit_if_json(run_payload(run_id, results))


@unlearning_app.command("list", epilog=hirundo_epilog)
def unlearning_list(
    archived: ArchivedOption = False,
    output: OutputOption = OutputFormat.text,
):
    """
    List LLM unlearning runs.
    """
    set_output_format(output)
    from hirundo.unlearning_llm import LlmUnlearningRun

    runs = LlmUnlearningRun.list(archived=archived)
    items = [
        {
            "name": str(run.name),
            "run_id": str(run.run_id),
            "status": str(run.status),
            "created_at": run.created_at.isoformat(),
        }
        for run in runs
    ]
    emit_rows(
        "Unlearning Runs:",
        [
            ("Name", "name"),
            ("Run ID", "run_id"),
            ("Status", "status"),
            ("Created At", "created_at"),
        ],
        items,
    )


@unlearning_app.command("check", epilog=hirundo_epilog)
def unlearning_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
    output: OutputOption = OutputFormat.text,
):
    """
    Check the status of an LLM unlearning run and stream progress.
    """
    set_output_format(output)
    from hirundo.unlearning_llm import LlmUnlearningRun

    results = check_run_and_print(run_id, LlmUnlearningRun.check_run_by_id)
    emit_if_json(run_payload(run_id, results))
