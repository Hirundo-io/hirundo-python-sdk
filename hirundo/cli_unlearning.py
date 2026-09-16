from typing import Annotated

import typer

from hirundo._cli_common import (
    ArchivedOption,
    WaitOption,
    check_run_and_print,
    hirundo_epilog,
    make_app,
    print_runs_table,
    report_run_started,
    require_exactly_one,
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
):
    """
    Launch an LLM unlearning run.

    Exactly one behavior option must be provided.
    """
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
        target_behavior = RefusalBehavior()
    else:  # unreachable: require_exactly_one guarantees one is set
        raise typer.Exit(code=1) from None

    run_info = LlmRunInfo(
        name=name,
        target_behaviors=[target_behavior],
    )

    run_id = LlmUnlearningRun.launch(model_id, run_info)
    report_run_started("Unlearning", run_id)

    wait_or_notify(run_id, LlmUnlearningRun.check_run_by_id, "unlearning", wait)


@unlearning_app.command("list", epilog=hirundo_epilog)
def unlearning_list(archived: ArchivedOption = False):
    """
    List LLM unlearning runs.
    """
    from hirundo.unlearning_llm import LlmUnlearningRun

    runs = LlmUnlearningRun.list(archived=archived)
    print_runs_table(
        "Unlearning Runs:",
        ("Name", "Run ID", "Status", "Created At"),
        [
            (
                str(run.name),
                str(run.run_id),
                str(run.status),
                run.created_at.isoformat(),
            )
            for run in runs
        ],
    )


@unlearning_app.command("check", epilog=hirundo_epilog)
def unlearning_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of an LLM unlearning run and stream progress.
    """
    from hirundo.unlearning_llm import LlmUnlearningRun

    check_run_and_print(run_id, LlmUnlearningRun.check_run_by_id)
