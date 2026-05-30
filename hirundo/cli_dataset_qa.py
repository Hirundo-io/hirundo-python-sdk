import json
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
    run_payload,
    set_output_format,
    wait_or_notify,
)

dataset_qa_app = make_app("dataset-qa", "Launch and monitor Dataset QA runs.")


@dataset_qa_app.command("run", epilog=hirundo_epilog)
def dataset_qa_run(
    dataset_id: Annotated[int, typer.Argument(help="ID of the dataset to run QA on.")],
    wait: WaitOption = True,
    output: OutputOption = OutputFormat.text,
):
    """
    Launch a Dataset QA run on the dataset with the given ID.
    """
    set_output_format(output)
    from hirundo.dataset_qa import QADataset

    run_id = QADataset.launch_qa_run(dataset_id)
    report_run_started("Dataset QA", run_id)

    results = wait_or_notify(run_id, QADataset.check_run_by_id, "dataset-qa", wait)
    if is_json():
        emit_json(run_payload(run_id, results))


@dataset_qa_app.command("list", epilog=hirundo_epilog)
def dataset_qa_list(
    archived: ArchivedOption = False,
    output: OutputOption = OutputFormat.text,
):
    """
    List Dataset QA runs.
    """
    set_output_format(output)
    from hirundo.dataset_qa import QADataset

    runs = QADataset.list_runs(archived=archived)
    items = [
        {
            "dataset_name": str(run.name),
            "run_id": str(run.run_id),
            "status": str(run.status),
            "created_at": run.created_at.isoformat(),
            "run_args": run.run_args.model_dump(mode="json") if run.run_args else None,
        }
        for run in runs
    ]
    emit(
        items,
        lambda: print_runs_table(
            "Dataset QA Runs:",
            ("Dataset Name", "Run ID", "Status", "Created At", "Run Args"),
            [
                (
                    item["dataset_name"],
                    item["run_id"],
                    item["status"],
                    item["created_at"],
                    json.dumps(item["run_args"]) if item["run_args"] else None,
                )
                for item in items
            ],
        ),
    )


@dataset_qa_app.command("check", epilog=hirundo_epilog)
def dataset_qa_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
    output: OutputOption = OutputFormat.text,
):
    """
    Check the status of a Dataset QA run and stream progress.
    """
    set_output_format(output)
    from hirundo.dataset_qa import QADataset

    results = check_run_and_print(run_id, QADataset.check_run_by_id)
    if is_json():
        emit_json(run_payload(run_id, results))
