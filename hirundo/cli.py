import os
import re
import sys
import typing
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from hirundo._env import API_HOST, EnvLocation

docs = "sphinx" in sys.modules
hirundo_epilog = (
    None
    if docs
    else "Made with ❤️ by Hirundo. Visit https://www.hirundo.io for more information."
)


app = typer.Typer(
    name="hirundo",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=hirundo_epilog,
)


def _upsert_env(
    dotenv_filepath: typing.Union[str, Path], var_name: str, var_value: str
):
    """
    Change an environment variable in the .env file.
    If the variable does not exist, it will be added.

    Args:
        var_name: The name of the environment variable to change.
        var_value: The new value of the environment variable.
    """
    regex = re.compile(rf"^{var_name}=.*$")
    lines = []
    if os.path.exists(dotenv_filepath):
        with open(dotenv_filepath) as f:
            lines = f.readlines()

    with open(dotenv_filepath, "w") as f:
        f.writelines(line for line in lines if not regex.search(line) and line != "\n")

    with open(dotenv_filepath, "a") as f:
        f.writelines(f"\n{var_name}={var_value}")


def upsert_env(var_name: str, var_value: str):
    if os.path.exists(EnvLocation.DOTENV.value):
        # If a `.env` file exists, re-use it
        _upsert_env(EnvLocation.DOTENV.value, var_name, var_value)
        return EnvLocation.DOTENV.name
    else:
        # Create a `.hirundo.conf` file with environment variables in the home directory
        _upsert_env(EnvLocation.HOME.value, var_name, var_value)
        return EnvLocation.HOME.name


def fix_api_host(api_host: str):
    if not api_host.startswith("http") and not api_host.startswith("https"):
        api_host = f"https://{api_host}"
        print(
            "API host must start with 'http://' or 'https://'. Automatically added 'https://'."
        )
    if (url := urlparse(api_host)) and url.path != "":
        print("API host should not contain a path. Removing path.")
        api_host = f"{url.scheme}://{url.hostname}"
    return api_host


@app.command("set-api-key", epilog=hirundo_epilog)
def setup_api_key(
    api_key: Annotated[
        str,
        typer.Option(
            prompt="Please enter the API key value",
            help=""
            if docs
            else f"Visit '{API_HOST}/api-key' to generate your API key.",
        ),
    ],
):
    """
    Setup the API key for the Hirundo Python SDK.
    Values are saved to a .env file in the current directory for use by the library in requests.
    """
    saved_to = upsert_env("API_KEY", api_key)
    if saved_to == EnvLocation.HOME.name:
        print(
            "API key saved to ~/.hirundo.conf for future use. Please do not share the ~/.hirundo.conf file since it contains your secret API key."
        )
    elif saved_to == EnvLocation.DOTENV.name:
        print(
            "API key saved to local .env file for future use. Please do not share the .env file since it contains your secret API key."
        )


@app.command("change-remote", epilog=hirundo_epilog)
def change_api_remote(
    api_host: Annotated[
        str,  # TODO: Change to HttpUrl when https://github.com/tiangolo/typer/pull/723 is merged
        typer.Option(
            prompt="Please enter the API server address",
            help=""
            if docs
            else f"Current API server address: '{API_HOST}'. This is the same address where you access the Hirundo web interface.",
        ),
    ],
):
    """
    Change the API server address for the Hirundo Python SDK.
    This is the same address where you access the Hirundo web interface.
    """
    api_host = fix_api_host(api_host)

    saved_to = upsert_env("API_HOST", api_host)
    if saved_to == EnvLocation.HOME.name:
        print(
            "API host saved to ~/.hirundo.conf for future use. Please do not share the ~/.hirundo.conf file"
        )
    elif saved_to == EnvLocation.DOTENV.name:
        print("API host saved to .env for future use. Please do not share this file")


@app.command("setup", epilog=hirundo_epilog)
def setup(
    api_key: Annotated[
        str,
        typer.Option(
            prompt="Please enter the API key value",
            help=""
            if docs
            else f"Visit '{API_HOST}/api-key' to generate your API key.",
        ),
    ],
    api_host: Annotated[
        str,  # TODO: Change to HttpUrl as above
        typer.Option(
            prompt="Please enter the API server address",
            help=""
            if docs
            else f"Current API server address: '{API_HOST}'. This is the same address where you access the Hirundo web interface.",
        ),
    ],
):
    """
    Setup the Hirundo Python SDK.
    """
    api_host = fix_api_host(api_host)
    api_host_saved_to = upsert_env("API_HOST", api_host)
    api_key_saved_to = upsert_env("API_KEY", api_key)
    if api_host_saved_to != api_key_saved_to:
        print(
            "API host and API key saved to different locations. This should not happen. Please report this issue."
        )
        if (
            api_host_saved_to == EnvLocation.HOME.name
            and api_key_saved_to == EnvLocation.DOTENV.name
        ):
            print(
                "API host saved to ~/.hirundo.conf for future use. Please do not share the ~/.hirundo.conf file"
            )
            print(
                "API key saved to local .env file for future use. Please do not share the .env file since it contains your secret API key."
            )
        elif (
            api_host_saved_to == EnvLocation.DOTENV.name
            and api_key_saved_to == EnvLocation.HOME.name
        ):
            print(
                "API host saved to .env for future use. Please do not share this file"
            )
            print(
                "API key saved to ~/.hirundo.conf for future use. Please do not share the ~/.hirundo.conf file since it contains your secret API key."
            )
        return
    if api_host_saved_to == EnvLocation.HOME.name:
        print(
            "API host and API key saved to ~/.hirundo.conf for future use. Please do not share the ~/.hirundo.conf file since it contains your secret API key."
        )
    elif api_host_saved_to == EnvLocation.DOTENV.name:
        print(
            "API host and API key saved to .env for future use. Please do not share this file since it contains your secret API key."
        )


@app.command("check-run", epilog=hirundo_epilog)
def check_run(
    run_id: str,
):
    """
    Check the status of a run.
    """
    from hirundo.dataset_qa import QADataset

    results = QADataset.check_run_by_id(run_id)
    print(f"Run results saved to {results.cached_zip_path}")


@app.command("list-runs", epilog=hirundo_epilog)
def list_runs():
    """
    List all runs available.
    """
    from hirundo.dataset_qa import QADataset

    runs = QADataset.list_runs()

    console = Console()
    table = Table(
        title="Runs:",
        expand=True,
    )
    cols = (
        "Dataset name",
        "Run ID",
        "Status",
        "Created At",
        "Run Args",
    )
    for col in cols:
        table.add_column(
            col,
            overflow="fold",
        )
    for run in runs:
        table.add_row(
            str(run.name),
            str(run.id),
            str(run.status),
            run.created_at.isoformat(),
            run.run_args.model_dump_json() if run.run_args else None,
        )
    console.print(table)


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
