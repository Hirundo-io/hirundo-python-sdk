import os
import re
from pathlib import Path
from typing import Annotated, TypeAlias
from urllib.parse import urlparse

import typer

from hirundo._cli_common import (
    HirundoCliGroup,
    OutputFormat,
    OutputOption,
    docs,
    emit_json,
    hirundo_epilog,
    is_json,
    set_output_format,
    success,
    warn,
)
from hirundo._env import API_HOST, EnvLocation
from hirundo.cli_dataset_qa import dataset_qa_app, dataset_qa_check, dataset_qa_list
from hirundo.cli_eval import eval_app
from hirundo.cli_unlearning import unlearning_app

_CONFIG_PANEL = "Configuration"
_RUNS_PANEL = "Runs"
_PIPELINES_PANEL = "Pipelines"

app = typer.Typer(
    name="hirundo",
    cls=HirundoCliGroup,
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
    # Re-use a local `.env` if present, otherwise fall back to `~/.hirundo.conf`.
    location = (
        EnvLocation.DOTENV
        if os.path.exists(EnvLocation.DOTENV.value)
        else EnvLocation.HOME
    )
    _upsert_env(location.value, var_name, var_value)
    return location.name


# Shared option definitions reused across set-api-key, change-remote, and setup.
_API_KEY_OPTION: TypeAlias = Annotated[
    str,
    typer.Option(
        prompt="Please enter the API key value",
        help="" if docs else f"Visit '{API_HOST}/api-key' to generate your API key.",
    ),
]

# TODO: Change to HttpUrl when https://github.com/tiangolo/typer/pull/723 is merged
_API_HOST_OPTION: TypeAlias = Annotated[
    str,
    typer.Option(
        prompt="Please enter the API server address",
        help=""
        if docs
        else (
            f"Current API server address: '{API_HOST}'. "
            "This is the same address where you access the Hirundo web interface."
        ),
    ),
]


def fix_api_host(api_host: str):
    if not api_host.startswith(("http://", "https://")):
        api_host = f"https://{api_host}"
        warn("API host must start with 'http://' or 'https://'. Added 'https://'.")
    if (url := urlparse(api_host)) and url.path != "":
        warn("API host should not contain a path. Removing it.")
        api_host = f"{url.scheme}://{url.hostname}"
    return api_host


def _save_api_key(api_key: str) -> str:
    location = _location_label(upsert_env("API_KEY", api_key))
    success(f"API key saved to [bold]{location}[/bold].")
    warn(f"Keep [bold]{location}[/bold] private — it contains your secret API key.")
    return location


def _save_api_host(api_host: str) -> str:
    location = _location_label(upsert_env("API_HOST", fix_api_host(api_host)))
    success(f"API host saved to [bold]{location}[/bold].")
    return location


@app.command("set-api-key", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
def setup_api_key(
    api_key: _API_KEY_OPTION,
    output: OutputOption = OutputFormat.text,
):
    """
    Save the API key for the Hirundo SDK.

    The key is written to a local .env file (or ~/.hirundo.conf if no .env
    exists) and picked up automatically on subsequent commands.
    """
    set_output_format(output)
    location = _save_api_key(api_key)
    if is_json():
        emit_json({"api_key_saved_to": location})


@app.command("change-remote", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
def change_api_remote(
    api_host: _API_HOST_OPTION,
    output: OutputOption = OutputFormat.text,
):
    """
    Change the API server address (same URL as the Hirundo web interface).
    """
    set_output_format(output)
    location = _save_api_host(api_host)
    if is_json():
        emit_json({"api_host_saved_to": location})


@app.command("setup", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
def setup(
    api_key: _API_KEY_OPTION,
    api_host: _API_HOST_OPTION,
    output: OutputOption = OutputFormat.text,
):
    """
    Setup the Hirundo Python SDK.
    """
    set_output_format(output)
    host_location = _save_api_host(api_host)
    key_location = _save_api_key(api_key)
    if is_json():
        emit_json(
            {"api_host_saved_to": host_location, "api_key_saved_to": key_location}
        )


@app.command("check-run", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def check_run(
    run_id: str,
    output: OutputOption = OutputFormat.text,
):
    """
    Check the status of a run.
    """
    dataset_qa_check(run_id, output)


@app.command("list-runs", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def list_runs(output: OutputOption = OutputFormat.text):
    """
    List all runs available.
    """
    dataset_qa_list(archived=False, output=output)


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
