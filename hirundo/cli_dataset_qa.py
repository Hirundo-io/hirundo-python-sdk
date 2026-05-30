from typing import Annotated

import typer

from hirundo._cli_common import (
    ArchivedOption,
    check_run_and_print,
    hirundo_epilog,
    make_app,
    print_runs_table,
    report_run_started,
    wait_or_notify,
)

dataset_qa_app = make_app("dataset-qa", "Launch and monitor Dataset QA runs.")


@dataset_qa_app.command("run", epilog=hirundo_epilog)
def dataset_qa_run(
    dataset_id: Annotated[int, typer.Argument(help="ID of the dataset to run QA on.")],
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait", help="Wait for the run to complete and stream progress."
        ),
    ] = True,
):
    """
    Launch a Dataset QA run on the dataset with the given ID.
    """
    from hirundo.dataset_qa import QADataset

    run_id = QADataset.launch_qa_run(dataset_id)
    report_run_started("Dataset QA", run_id)

    wait_or_notify(run_id, QADataset.check_run_by_id, "dataset-qa", wait)


@dataset_qa_app.command("list", epilog=hirundo_epilog)
def dataset_qa_list(archived: ArchivedOption = False):
    """
    List Dataset QA runs.
    """
    from hirundo.dataset_qa import QADataset

    runs = QADataset.list_runs(archived=archived)
    print_runs_table(
        "Dataset QA Runs:",
        ("Dataset Name", "Run ID", "Status", "Created At", "Run Args"),
        [
            (
                str(run.name),
                str(run.run_id),
                str(run.status),
                run.created_at.isoformat(),
                run.run_args.model_dump_json() if run.run_args else None,
            )
            for run in runs
        ],
    )


@dataset_qa_app.command("check", epilog=hirundo_epilog)
def dataset_qa_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of a Dataset QA run and stream progress.
    """
    from hirundo.dataset_qa import QADataset

    check_run_and_print(run_id, QADataset.check_run_by_id)
