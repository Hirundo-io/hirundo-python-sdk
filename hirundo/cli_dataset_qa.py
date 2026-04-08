"""
CLI sub-app for Dataset QA commands.

Commands:
    hirundo dataset-qa run    - Launch a Dataset QA run
    hirundo dataset-qa list   - List Dataset QA runs
    hirundo dataset-qa check  - Check the status of a Dataset QA run
"""

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

docs = "sphinx" in sys.modules
dataset_qa_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)

console = Console()

dataset_qa_app = typer.Typer(
    name="dataset-qa",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=dataset_qa_epilog,
    help="Launch and monitor Dataset QA runs.",
)


@dataset_qa_app.command("run", epilog=dataset_qa_epilog)
def dataset_qa_run(
    dataset_id: Annotated[int, typer.Argument(help="ID of the dataset to run QA on.")],
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait for the run to complete and stream progress."),
    ] = True,
):
    """
    Launch a Dataset QA run on the dataset with the given ID.
    """
    from hirundo.dataset_qa import QADataset

    run_id = QADataset.launch_qa_run(dataset_id)
    console.print(f"Dataset QA run started. Run ID: [bold]{run_id}[/bold]")

    if wait:
        QADataset.check_run_by_id(run_id)
    else:
        console.print(
            "Use [bold]hirundo dataset-qa check[/bold] [italic]<run_id>[/italic] to monitor progress."
        )


@dataset_qa_app.command("list", epilog=dataset_qa_epilog)
def dataset_qa_list(
    archived: Annotated[
        bool,
        typer.Option("--archived/--no-archived", help="Include archived runs."),
    ] = False,
):
    """
    List Dataset QA runs.
    """
    from hirundo.dataset_qa import QADataset

    runs = QADataset.list_runs(archived=archived)

    table = Table(title="Dataset QA Runs:", expand=True)
    for col in ("Dataset Name", "Run ID", "Status", "Created At", "Run Args"):
        table.add_column(col, overflow="fold")
    for run in runs:
        table.add_row(
            str(run.name),
            str(run.run_id),
            str(run.status),
            run.created_at.isoformat(),
            run.run_args.model_dump_json() if run.run_args else None,
        )
    console.print(table)


@dataset_qa_app.command("check", epilog=dataset_qa_epilog)
def dataset_qa_check(
    run_id: Annotated[str, typer.Argument(help="The run ID to check.")],
):
    """
    Check the status of a Dataset QA run and stream progress.
    """
    from hirundo.dataset_qa import QADataset

    QADataset.check_run_by_id(run_id)
