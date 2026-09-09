"""Publish and verify sanitized VCR.py cassette artifacts.

Run with ``python -m tests.recording.artifact_cli --help``. The command never
accepts secret values as command-line arguments because process arguments may be
captured in CI logs.
"""

import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import typer
import yaml
from tests.recording.support import (
    Cassette,
    JsonValue,
    RecordingManifest,
    UnsafeCassetteError,
    as_cassette,
    as_json_value,
    sanitize_cassette,
)

PILOT_RECORD_NAME_PREFIX = "sdk-http-recording-"

app = typer.Typer(add_completion=False)


def _secrets(*, environment_name: str | None, secrets_file: Path | None) -> list[str]:
    values: list[str] = []
    if environment_name:
        environment_value = os.environ.get(environment_name)
        if environment_value:
            values.extend(environment_value.splitlines())
    if secrets_file:
        values.extend(secrets_file.read_text(encoding="utf-8").splitlines())
    return [value for value in values if value]


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("PyYAML", "vcrpy", "pytest-recording"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _response_json(response: Cassette) -> JsonValue:
    body = response.get("body")
    raw_body = body.get("string") if isinstance(body, dict) else body
    if isinstance(raw_body, bytes):
        try:
            raw_body = raw_body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UnsafeCassetteError("git-repo list response is not UTF-8") from error
    if not isinstance(raw_body, str):
        raise UnsafeCassetteError("git-repo list response has no JSON body")
    try:
        loaded: object = json.loads(raw_body)
        return as_json_value(loaded)
    except (json.JSONDecodeError, ValueError) as error:
        raise UnsafeCassetteError("git-repo list response is not JSON") from error


def _replace_response_json(response: Cassette, value: JsonValue) -> None:
    serialized = json.dumps(value, separators=(",", ":"))
    serialized_bytes = serialized.encode("utf-8")
    body = response.get("body")
    if isinstance(body, dict):
        body["string"] = (
            serialized_bytes if isinstance(body.get("string"), bytes) else serialized
        )
    else:
        response["body"] = serialized_bytes if isinstance(body, bytes) else serialized
    headers = response.get("headers")
    if isinstance(headers, dict):
        for name, header_value in headers.items():
            if name.lower() == "content-length":
                headers[name] = (
                    [str(len(serialized_bytes))]
                    if isinstance(header_value, list)
                    else str(len(serialized_bytes))
                )


def _filter_git_repo_response(response: Cassette) -> None:
    payload = _response_json(response)
    if not isinstance(payload, list):
        raise UnsafeCassetteError("git-repo list response is not a JSON list")
    filtered_records: list[JsonValue] = []
    for record in payload:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise UnsafeCassetteError("git-repo list contains a malformed record")
        record_name = record["name"]
        if isinstance(record_name, str) and record_name.startswith(
            PILOT_RECORD_NAME_PREFIX
        ):
            filtered_records.append(record)
    _replace_response_json(response, filtered_records)


def apply_pilot_privacy_policy(cassette: Cassette) -> None:
    """Remove unrelated repositories from pilot list responses.

    Args:
        cassette: The sanitized VCR cassette to filter in place.

    Returns:
        None.
    """

    interactions = cassette.get("interactions")
    if not isinstance(interactions, list):
        raise UnsafeCassetteError("cassette interactions must be a list")
    for interaction in interactions:
        if not isinstance(interaction, dict):
            raise UnsafeCassetteError("cassette interaction must be a mapping")
        request = interaction["request"]
        response = interaction["response"]
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise UnsafeCassetteError(
                "interaction request and response must be mappings"
            )
        request_url = request.get("uri", request.get("url", ""))
        if not isinstance(request_url, str):
            raise UnsafeCassetteError("cassette request URL must be text")
        request_path = urlsplit(request_url).path
        if (
            str(request.get("method", "")).upper() != "GET"
            or request_path.rstrip("/").split("/")[-1] != "git-repo"
        ):
            continue
        _filter_git_repo_response(response)


def publish_cassettes(
    *,
    raw_directory: Path,
    publish_directory: Path,
    seeded_secrets: list[str],
    sdk_sha: str,
    run_id: str,
    run_attempt: int,
    environment: str,
    recording_started_at: str,
    recording_finished_at: str,
    expires_at: str,
    schema_digest: str,
    test_selection: tuple[str, ...],
) -> RecordingManifest:
    """Sanitize all raw YAML cassettes and write their replay manifest.

    Args:
        raw_directory: Root directory containing raw cassette YAML files.
        publish_directory: Empty destination for sanitized artifacts.
        seeded_secrets: Secret markers that must not appear in published files.
        sdk_sha: SDK commit that produced the recordings.
        run_id: CI run identifier that produced the recordings.
        run_attempt: CI run attempt that produced the recordings.
        environment: Name of the API environment used for recording.
        recording_started_at: UTC ISO 8601 recording start time.
        recording_finished_at: UTC ISO 8601 recording finish time.
        expires_at: UTC ISO 8601 artifact expiry time.
        schema_digest: SHA-256 digest of the API schema.
        test_selection: Pytest node IDs included in the recording job.

    Returns:
        The validated manifest written beside the sanitized cassettes.
    """

    if raw_directory.resolve() == publish_directory.resolve():
        raise ValueError("raw and publish directories must differ")
    if not seeded_secrets:
        raise ValueError("at least one seeded secret is required")
    raw_paths = sorted(raw_directory.rglob("*.yaml"))
    if not raw_paths:
        raise ValueError("raw directory contains no YAML cassettes")
    if publish_directory.exists() and any(publish_directory.iterdir()):
        raise ValueError("publish directory must be empty")
    sanitized_documents: list[tuple[Path, str]] = []
    interaction_count = 0
    for raw_path in raw_paths:
        if raw_path.is_symlink():
            raise ValueError("cassette symlinks are not allowed")
        relative_path = raw_path.relative_to(raw_directory)
        loaded: object = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
        try:
            raw_cassette = as_cassette(loaded)
        except ValueError as error:
            raise ValueError(
                f"cassette {relative_path} must contain a mapping"
            ) from error
        sanitized = sanitize_cassette(raw_cassette, seeded_secrets=seeded_secrets)
        apply_pilot_privacy_policy(sanitized)
        sanitized_interactions = sanitized["interactions"]
        if not isinstance(sanitized_interactions, list):
            raise UnsafeCassetteError("cassette interactions must be a list")
        interaction_count += len(sanitized_interactions)
        sanitized_documents.append(
            (
                relative_path,
                yaml.safe_dump(sanitized, allow_unicode=True, sort_keys=True),
            )
        )

    publish_directory.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for relative_path, sanitized_yaml in sanitized_documents:
        publish_path = publish_directory / relative_path
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        publish_path.write_text(sanitized_yaml, encoding="utf-8")
        checksums[relative_path.as_posix()] = _checksum(publish_path)

    manifest = RecordingManifest(
        sdk_sha=sdk_sha,
        run_id=run_id,
        run_attempt=run_attempt,
        environment=environment,
        recording_started_at=recording_started_at,
        recording_finished_at=recording_finished_at,
        expires_at=expires_at,
        schema_digest=schema_digest,
        platform_revision="unknown",
        tool_versions=_tool_versions(),
        test_selection=test_selection,
        cassette_format="vcrpy-yaml-v1",
        interaction_count=interaction_count,
        artifact_bytes=sum(
            len(sanitized_yaml.encode("utf-8"))
            for _, sanitized_yaml in sanitized_documents
        ),
        cassette_checksums=checksums,
    )
    manifest.validate(publish_directory)
    manifest_path = publish_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_cassettes(
    *,
    publish_directory: Path,
    expected_sdk_sha: str,
    expected_run_id: str,
    expected_run_attempt: int,
    now: datetime | None = None,
) -> RecordingManifest:
    """Verify artifact provenance and contents before cassette replay.

    Args:
        publish_directory: Directory containing the manifest and cassettes.
        expected_sdk_sha: Exact SDK commit expected by the replay job.
        expected_run_id: Exact CI run identifier expected by the replay job.
        expected_run_attempt: Exact CI attempt expected by the replay job.
        now: UTC time used for expiry validation, or the current time when omitted.

    Returns:
        The verified recording manifest.
    """

    loaded: object = json.loads(
        (publish_directory / "manifest.json").read_text(encoding="utf-8")
    )
    try:
        manifest_mapping = as_cassette(loaded)
    except ValueError as error:
        raise ValueError("manifest must contain an object") from error
    manifest = RecordingManifest.from_mapping(manifest_mapping)
    manifest.verify_replay(
        cassette_root=publish_directory,
        expected_sdk_sha=expected_sdk_sha,
        expected_run_id=expected_run_id,
        expected_run_attempt=expected_run_attempt,
        now=now,
    )
    return manifest


@app.command("publish")
def publish_command(
    raw_directory: Annotated[
        Path, typer.Argument(help="Directory containing raw VCR YAML files.")
    ],
    publish_directory: Annotated[
        Path, typer.Argument(help="Empty destination for sanitized artifacts.")
    ],
    sdk_sha: Annotated[str, typer.Option("--sdk-sha")],
    run_id: Annotated[str, typer.Option("--run-id")],
    run_attempt: Annotated[int, typer.Option("--run-attempt")],
    environment: Annotated[str, typer.Option("--environment")],
    recording_started_at: Annotated[str, typer.Option("--recording-started-at")],
    recording_finished_at: Annotated[str, typer.Option("--recording-finished-at")],
    expires_at: Annotated[str, typer.Option("--expires-at")],
    schema_digest: Annotated[str, typer.Option("--schema-digest")],
    tests: Annotated[list[str], typer.Option("--test")],
    secrets_env: Annotated[str | None, typer.Option("--secrets-env")] = None,
    secrets_file: Annotated[Path | None, typer.Option("--secrets-file")] = None,
) -> None:
    """Sanitize raw cassettes and publish a provenance-bound artifact.

    Args:
        raw_directory: Root directory containing raw cassette YAML files.
        publish_directory: Empty destination for sanitized artifacts.
        sdk_sha: SDK commit that produced the recordings.
        run_id: CI run identifier that produced the recordings.
        run_attempt: CI run attempt that produced the recordings.
        environment: Name of the API environment used for recording.
        recording_started_at: UTC ISO 8601 recording start time.
        recording_finished_at: UTC ISO 8601 recording finish time.
        expires_at: UTC ISO 8601 artifact expiry time.
        schema_digest: SHA-256 digest of the API schema.
        tests: Pytest node IDs included in the recording job.
        secrets_env: Environment variable containing newline-separated secrets.
        secrets_file: Optional file containing newline-separated secrets.

    Returns:
        None.
    """
    publish_cassettes(
        raw_directory=raw_directory,
        publish_directory=publish_directory,
        seeded_secrets=_secrets(
            environment_name=secrets_env,
            secrets_file=secrets_file,
        ),
        sdk_sha=sdk_sha,
        run_id=run_id,
        run_attempt=run_attempt,
        environment=environment,
        recording_started_at=recording_started_at,
        recording_finished_at=recording_finished_at,
        expires_at=expires_at,
        schema_digest=schema_digest,
        test_selection=tuple(tests),
    )


@app.command("verify")
def verify_command(
    publish_directory: Annotated[
        Path, typer.Argument(help="Directory containing published artifacts.")
    ],
    sdk_sha: Annotated[str, typer.Option("--sdk-sha")],
    run_id: Annotated[str, typer.Option("--run-id")],
    run_attempt: Annotated[int, typer.Option("--run-attempt")],
) -> None:
    """Verify artifact provenance, expiry, and checksums before replay.

    Args:
        publish_directory: Directory containing the manifest and cassettes.
        sdk_sha: Exact SDK commit expected by the replay job.
        run_id: Exact CI run identifier expected by the replay job.
        run_attempt: Exact CI attempt expected by the replay job.

    Returns:
        None.
    """
    verify_cassettes(
        publish_directory=publish_directory,
        expected_sdk_sha=sdk_sha,
        expected_run_id=run_id,
        expected_run_attempt=run_attempt,
        now=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    app()
