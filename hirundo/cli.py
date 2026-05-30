import os
import re
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer

from hirundo._cli_common import docs, hirundo_epilog, success, warn
from hirundo._env import API_HOST, EnvLocation
from hirundo.cli_dataset_qa import dataset_qa_app, dataset_qa_check, dataset_qa_list
from hirundo.cli_eval import eval_app
from hirundo.cli_unlearning import unlearning_app

_CONFIG_PANEL = "Configuration"
_RUNS_PANEL = "Runs"
_PIPELINES_PANEL = "Pipelines"

app = typer.Typer(
    name="hirundo",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=hirundo_epilog,
    help=(
        "Launch and monitor Hirundo data-quality, unlearning, and evaluation "
        "runs. Run `hirundo setup` once to store your API key, then use the "
        "eval, dataset-qa, and unlearning command groups."
    ),
)

app.add_typer(eval_app, name="eval", rich_help_panel=_PIPELINES_PANEL)
app.add_typer(dataset_qa_app, name="dataset-qa", rich_help_panel=_PIPELINES_PANEL)
app.add_typer(unlearning_app, name="unlearning", rich_help_panel=_PIPELINES_PANEL)


def _location_label(saved_to: str) -> str:
    """Human-friendly name of the file an env var was written to."""
    return "~/.hirundo.conf" if saved_to == EnvLocation.HOME.name else ".env"


def _upsert_env(dotenv_filepath: str | Path, var_name: str, var_value: str):
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
        warn("API host must start with 'http://' or 'https://'. Added 'https://'.")
    if (url := urlparse(api_host)) and url.path != "":
        warn("API host should not contain a path. Removing it.")
        api_host = f"{url.scheme}://{url.hostname}"
    return api_host


@app.command("set-api-key", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
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
    location = _location_label(upsert_env("API_KEY", api_key))
    success(f"API key saved to [bold]{location}[/bold].")
    warn(f"Keep [bold]{location}[/bold] private — it contains your secret API key.")


@app.command("change-remote", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
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

    location = _location_label(upsert_env("API_HOST", api_host))
    success(f"API host saved to [bold]{location}[/bold].")


@app.command("setup", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
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
    host_location = _location_label(upsert_env("API_HOST", api_host))
    key_location = _location_label(upsert_env("API_KEY", api_key))

    success(f"API host saved to [bold]{host_location}[/bold].")
    success(f"API key saved to [bold]{key_location}[/bold].")
    warn(f"Keep [bold]{key_location}[/bold] private — it contains your secret API key.")


@app.command("check-run", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def check_run(
    run_id: str,
):
    """
    Check the status of a run.
    """
    dataset_qa_check(run_id)


@app.command("list-runs", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def list_runs():
    """
    List all runs available.
    """
    dataset_qa_list(archived=False)


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
