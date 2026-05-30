import os
import re
import stat
import tempfile
from collections.abc import Callable
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Annotated, TypeAlias, cast
from urllib.parse import urlparse

import typer
from dotenv import dotenv_values, set_key, unset_key

from hirundo._cli_common import (
    HirundoCliGroup,
    OutputFormat,
    OutputOption,
    check_run_and_print,
    docs,
    emit_if_json,
    emit_rows,
    hirundo_epilog,
    run_payload,
    set_output_format,
    success,
    warn,
)
from hirundo._credentials import (
    KeyringUnavailableError,
    delete_api_key_from_keyring,
    normalize_api_host,
    save_api_key_to_keyring,
)
from hirundo._env import API_HOST, EnvLocation
from hirundo.cli_dataset_qa import dataset_qa_app
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


class RunType(str, Enum):
    LLM_UNLEARNING = "llm-unlearning"
    LLM_EVALUATION = "llm-evaluation"
    DATASET_QA = "dataset-qa"
    EXTERNAL_EVALUATION = "external-evaluation"


class KeyStorage(str, Enum):
    AUTO = "auto"
    KEYRING = "keyring"
    FILE = "file"


def _location_label(saved_to: str) -> str:
    """Return the display name for an environment-file location."""
    return "~/.hirundo.conf" if saved_to == EnvLocation.HOME.name else ".env"


def _validate_env_value(var_name: str, var_value: str) -> None:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var_name) is None:
        raise ValueError(f"Invalid environment variable name: {var_name!r}")
    if any(character in var_value for character in ("\r", "\n", "\0")):
        raise ValueError(f"{var_name} must not contain line breaks or null bytes")


def _read_secure_config(
    dotenv_filepath: str | Path,
) -> tuple[Path, os.stat_result | None, str]:
    dotenv_path = Path(dotenv_filepath).resolve(strict=False)
    open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(dotenv_path, open_flags)
    except FileNotFoundError:
        return dotenv_path, None, ""

    try:
        file_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError(
                f"Configuration path must be a regular file: {dotenv_path}"
            )
        if hasattr(os, "fchmod"):
            os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, encoding="utf-8") as config_file:
            file_descriptor = -1
            contents = config_file.read()
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
    return dotenv_path, file_status, contents


def _rewrite_env(
    dotenv_filepath: str | Path,
    mutation: Callable[[Path], None],
) -> None:
    unresolved_path = Path(dotenv_filepath)
    dotenv_path = unresolved_path.resolve(strict=False)
    lock_path = dotenv_path.with_name(f".{dotenv_path.name}.hirundo.lock")
    try:
        lock_descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError:
        raise RuntimeError(
            f"Configuration file is already being updated: {dotenv_path}"
        ) from None
    temporary_path: Path | None = None
    try:
        os.close(lock_descriptor)
        dotenv_path, original_status, contents = _read_secure_config(dotenv_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            prefix=".hirundo-",
            dir=dotenv_path.parent,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)
        temporary_path.chmod(0o600)
        mutation(temporary_path)
        _, current_status, current_contents = _read_secure_config(dotenv_path)
        if (
            (original_status is None) != (current_status is None)
            or (original_status is not None and current_status is not None)
            and (
                current_status.st_dev != original_status.st_dev
                or current_status.st_ino != original_status.st_ino
            )
            or current_contents != contents
        ):
            raise RuntimeError(
                f"Configuration file changed while updating it: {dotenv_path}"
            )

        os.replace(temporary_path, dotenv_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def _upsert_env(dotenv_filepath: str | Path, var_name: str, var_value: str) -> None:
    """Add or replace an environment variable in a private dotenv file."""
    _validate_env_value(var_name, var_value)

    def set_value(temporary_path: Path) -> None:
        set_key(temporary_path, var_name, var_value, quote_mode="always")

    _rewrite_env(dotenv_filepath, set_value)


def _preferred_env_location() -> EnvLocation:
    # Re-use a local `.env` if present, otherwise fall back to `~/.hirundo.conf`.
    return (
        EnvLocation.DOTENV
        if os.path.exists(EnvLocation.DOTENV.value)
        else EnvLocation.HOME
    )


def upsert_env(
    var_name: str, var_value: str, location: EnvLocation | None = None
) -> str:
    location = location or _preferred_env_location()
    _upsert_env(location.value, var_name, var_value)
    return location.name


def _remove_env_key(dotenv_filepath: str | Path, var_name: str) -> None:
    dotenv_path, _, contents = _read_secure_config(dotenv_filepath)
    if var_name not in dotenv_values(stream=StringIO(contents)):
        return

    def remove_value(temporary_path: Path) -> None:
        unset_key(temporary_path, var_name)

    _rewrite_env(dotenv_path, remove_value)


# Shared option definitions reused across set-api-key, change-remote, and setup.
_API_KEY_OPTION: TypeAlias = Annotated[
    str,
    typer.Option(
        prompt="Please enter the API key value",
        hide_input=True,
        help="" if docs else f"Visit '{API_HOST}/api-key' to generate your API key.",
    ),
]

_KEY_STORAGE_OPTION: TypeAlias = Annotated[
    KeyStorage,
    typer.Option(
        "--key-storage",
        help=(
            "Where to persist the API key. 'auto' prefers the operating-system "
            "keyring and falls back to a private configuration file."
        ),
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


def fix_api_host(api_host: str) -> str:
    original_api_host = api_host
    has_http_scheme = original_api_host.lower().startswith(("http://", "https://"))
    if not has_http_scheme:
        warn("API host must start with 'http://' or 'https://'. Added 'https://'.")
    url = urlparse(
        original_api_host if has_http_scheme else f"https://{original_api_host}"
    )
    if url.path not in {"", "/"}:
        warn("API host should not contain a path. Removing it.")
    return normalize_api_host(original_api_host)


def _save_api_key_to_file(api_key: str, env_location: EnvLocation | None = None) -> str:
    location = _location_label(upsert_env("HIRUNDO_API_KEY", api_key, env_location))
    success(f"API key saved to [bold]{location}[/bold].")
    warn(f"Keep [bold]{location}[/bold] private; it contains your secret API key.")
    return location


def _save_api_key(
    api_key: str,
    api_host: str,
    key_storage: KeyStorage,
    env_location: EnvLocation | None = None,
) -> str:
    if key_storage is not KeyStorage.FILE:
        try:
            backend_name = save_api_key_to_keyring(api_host, api_key)
        except KeyringUnavailableError:
            if key_storage is KeyStorage.KEYRING:
                raise typer.BadParameter(
                    "No usable operating-system keyring is available. Use "
                    "HIRUNDO_API_KEY for non-interactive environments or select "
                    "--key-storage file."
                ) from None
            warn(
                "No usable operating-system keyring is available; falling back "
                "to a private configuration file. For CI, containers, SSH "
                "sessions, and headless servers, prefer HIRUNDO_API_KEY."
            )
        else:
            location = env_location or _preferred_env_location()
            _remove_env_key(location.value, "HIRUNDO_API_KEY")
            success(
                "API key saved to the operating-system keyring "
                f"([bold]{backend_name}[/bold])."
            )
            return "keyring"

    location = _save_api_key_to_file(api_key, env_location)
    if key_storage is KeyStorage.FILE:
        delete_api_key_from_keyring(api_host)
    return location


def _save_api_host(api_host: str, env_location: EnvLocation | None = None) -> str:
    api_host = fix_api_host(api_host)
    location = _location_label(upsert_env("HIRUNDO_API_HOST", api_host, env_location))
    success(f"API host saved to [bold]{location}[/bold].")
    return location


@app.command("set-api-key", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
def setup_api_key(
    api_key: _API_KEY_OPTION,
    key_storage: _KEY_STORAGE_OPTION = KeyStorage.AUTO,
    output: OutputOption = OutputFormat.text,
):
    """
    Save the API key for the Hirundo SDK.

    The key is stored in the operating-system keyring when one is available.
    Headless environments fall back to a private configuration file.
    """
    set_output_format(output)
    location = _save_api_key(api_key, API_HOST, key_storage)
    emit_if_json({"api_key_saved_to": location})


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
    emit_if_json({"api_host_saved_to": location})


@app.command("setup", epilog=hirundo_epilog, rich_help_panel=_CONFIG_PANEL)
def setup(
    api_key: _API_KEY_OPTION,
    api_host: _API_HOST_OPTION,
    key_storage: _KEY_STORAGE_OPTION = KeyStorage.AUTO,
    output: OutputOption = OutputFormat.text,
):
    """
    Setup the Hirundo Python SDK.
    """
    set_output_format(output)
    _validate_env_value("HIRUNDO_API_HOST", api_host)
    _validate_env_value("HIRUNDO_API_KEY", api_key)
    env_location = _preferred_env_location()
    normalized_api_host = fix_api_host(api_host)
    host_location = _save_api_host(normalized_api_host, env_location)
    key_location = _save_api_key(
        api_key, normalized_api_host, key_storage, env_location
    )
    emit_if_json({"api_host_saved_to": host_location, "api_key_saved_to": key_location})


@app.command("check-run", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def check_run(
    run_id: str,
    run_type: Annotated[
        RunType,
        typer.Option("--run-type", "-t", help="Type of run to check."),
    ] = RunType.LLM_UNLEARNING,
    output: OutputOption = OutputFormat.text,
):
    """
    Check the status of a run.
    """
    set_output_format(output)
    if run_type is RunType.LLM_UNLEARNING:
        from hirundo.unlearning_llm import LlmUnlearningRun

        check_function = LlmUnlearningRun.check_run_by_id
    elif run_type is RunType.LLM_EVALUATION:
        from hirundo.llm_behavior_eval import LlmBehaviorEval

        check_function = LlmBehaviorEval.check_run_by_id
    elif run_type is RunType.EXTERNAL_EVALUATION:
        from hirundo.external_eval import ExternalEval

        check_function = ExternalEval.check_run_by_id
    else:
        from hirundo.dataset_qa import QADataset

        check_function = QADataset.check_run_by_id

    results = check_run_and_print(
        run_id,
        check_function,
        validate=run_type is not RunType.EXTERNAL_EVALUATION,
    )
    if run_type is RunType.LLM_UNLEARNING and output is OutputFormat.text:
        print(results)
    emit_if_json(run_payload(run_id, results))


@app.command("list-runs", epilog=hirundo_epilog, rich_help_panel=_RUNS_PANEL)
def list_runs(
    run_type: Annotated[
        RunType,
        typer.Option("--run-type", "-t", help="Type of runs to list."),
    ] = RunType.LLM_UNLEARNING,
    output: OutputOption = OutputFormat.text,
):
    """
    List all runs available.
    """
    set_output_format(output)
    if run_type is RunType.LLM_UNLEARNING:
        from hirundo.unlearning_llm import LlmUnlearningRun

        runs = LlmUnlearningRun.list()
    elif run_type is RunType.LLM_EVALUATION:
        from hirundo.llm_behavior_eval import LlmBehaviorEval

        runs = LlmBehaviorEval.list_runs()
    elif run_type is RunType.DATASET_QA:
        from hirundo.dataset_qa import QADataset

        runs = QADataset.list_runs()
    else:
        from hirundo.llm_behavior_eval import EvalFramework, LlmBehaviorEval

        runs = [
            run_record
            for run_record in LlmBehaviorEval.list_runs()
            if run_record.framework is EvalFramework.INSPECT_EVALS
        ]

    columns = [
        ("Name", "name"),
        ("Run ID", "run_id"),
        ("Status", "status"),
        ("Created At", "created_at"),
    ]
    items = []
    for run_record in runs:
        item = {
            "name": str(run_record.name),
            "run_id": str(run_record.run_id),
            "status": str(run_record.status),
            "created_at": run_record.created_at.isoformat(),
        }
        if run_type is RunType.DATASET_QA:
            from hirundo.dataset_qa import DataQARunOut

            dataset_qa_run = cast("DataQARunOut", run_record)
            item["run_args"] = (
                dataset_qa_run.run_args.model_dump(mode="json")
                if dataset_qa_run.run_args
                else None
            )
        items.append(item)

    if run_type is RunType.DATASET_QA:
        columns.append(("Run Args", "run_args"))
    emit_rows("Runs:", columns, items)


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
