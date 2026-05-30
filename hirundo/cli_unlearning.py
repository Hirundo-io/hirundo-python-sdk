from typing import Annotated

import typer

from hirundo._cli_common import (
    ArchivedOption,
    OutputFormat,
    OutputOption,
    WaitOption,
    check_run_and_print,
    emit,
    emit_json,
    hirundo_epilog,
    is_json,
    make_app,
    print_runs_table,
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
    bias: Annotated[
        bool,
        typer.Option("--bias", help="Run bias unlearning."),
    ] = False,
    hallucination_type: Annotated[
        str | None,
        typer.Option(
            "--hallucination-type",
            help="Hallucination type for unlearning. One of: GENERAL, MEDICAL, LEGAL, DEFENSE",
        ),
    ] = None,
    security: Annotated[
        bool,
        typer.Option("--security", help="Run security unlearning."),
    ] = False,
    refusal: Annotated[
        bool,
        typer.Option("--refusal", help="Run refusal unlearning."),
    ] = False,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Optional name for this unlearning run."),
    ] = None,
    wait: WaitOption = True,
    output: OutputOption = OutputFormat.text,
):
    """
    Launch an LLM unlearning run.

    Exactly one behavior option must be provided.
    """
    set_output_format(output)
    from hirundo.unlearning_llm import (
        BiasBehavior,
        HallucinationBehavior,
        HallucinationType,
        LlmRunInfo,
        LlmUnlearningRun,
        RefusalBehavior,
        SecurityBehavior,
    )

    require_exactly_one(
        ("--bias", True if bias else None),
        ("--hallucination-type", hallucination_type),
        ("--security", True if security else None),
        ("--refusal", True if refusal else None),
    )

    if bias:
        target_behavior = BiasBehavior()
    elif hallucination_type is not None:
        target_behavior = HallucinationBehavior(
            hallucination_type=validate_enum(
                hallucination_type, HallucinationType, "hallucination type"
            )
        )
    elif security:
        target_behavior = SecurityBehavior()
    elif refusal:
        if not LlmUnlearningRun.get_capabilities().refusal_unlearning_enabled:
            raise typer.BadParameter(
                "Refusal unlearning is not enabled by the configured Hirundo API.",
                param_hint="--refusal",
            )
        target_behavior = RefusalBehavior()
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
    if is_json():
        emit_json(run_payload(run_id, results))


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
    emit(
        items,
        lambda: print_runs_table(
            "Unlearning Runs:",
            ("Name", "Run ID", "Status", "Created At"),
            [
                (item["name"], item["run_id"], item["status"], item["created_at"])
                for item in items
            ],
        ),
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
    if is_json():
        emit_json(run_payload(run_id, results))
