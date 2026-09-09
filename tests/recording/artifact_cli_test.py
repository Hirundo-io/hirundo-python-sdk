from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
import yaml
from tests.recording.artifact_cli import _secrets, publish_cassettes, verify_cassettes
from tests.recording.support import (
    Cassette,
    JsonValue,
    ManifestValidationError,
    UnsafeCassetteError,
    as_cassette,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def _raw_cassette(secret: str) -> Cassette:
    return {
        "version": 1,
        "interactions": [
            {
                "request": {
                    "method": "GET",
                    "uri": f"https://api.example.test/jobs?token={secret}",
                    "headers": {"Authorization": [f"Bearer {secret}"]},
                    "body": None,
                },
                "response": {
                    "status": {"code": 200, "message": "OK"},
                    "headers": {"Content-Type": ["application/json"]},
                    "body": {"string": json.dumps({"token": secret, "ok": True})},
                },
            }
        ],
    }


def _first_exchange(recorded: Cassette) -> tuple[Cassette, Cassette]:
    interactions = recorded["interactions"]
    assert isinstance(interactions, list)
    interaction = interactions[0]
    assert isinstance(interaction, dict)
    request = interaction["request"]
    response = interaction["response"]
    assert isinstance(request, dict)
    assert isinstance(response, dict)
    return request, response


def _git_repo_cassette(payload: JsonValue, *, body_as_bytes: bool = False) -> Cassette:
    recorded = _raw_cassette("marker")
    request, response = _first_exchange(recorded)
    request["uri"] = "https://api.example.test/api/git-repo/"
    serialized = json.dumps(payload)
    response["body"] = {
        "string": serialized.encode("utf-8") if body_as_bytes else serialized
    }
    response["headers"] = {
        "Content-Type": ["application/json"],
        "Content-Length": ["9999"],
    }
    return recorded


def _publish(raw_directory: Path, publish_directory: Path, secret: str) -> None:
    publish_cassettes(
        raw_directory=raw_directory,
        publish_directory=publish_directory,
        seeded_secrets=[secret],
        sdk_sha="a" * 40,
        run_id="98765",
        run_attempt=1,
        environment="designated-test",
        recording_started_at="2026-09-09T10:00:00Z",
        recording_finished_at="2026-09-09T10:05:00Z",
        expires_at="2026-09-10T10:00:00Z",
        schema_digest="sha256:" + "b" * 64,
        test_selection=("tests/pilot_test.py::test_download",),
    )


def test_secret_environment_values_preserve_multiline_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    credential_value = "first-line\nsecond-line"
    monkeypatch.setenv("MULTILINE_SECRET", credential_value)
    monkeypatch.setenv("TOKEN_SECRET", "single-line-token")

    assert _secrets(
        environment_names=["MULTILINE_SECRET", "TOKEN_SECRET"],
        secrets_file=None,
    ) == [credential_value, "single-line-token"]


def test_publish_sanitizes_all_yaml_and_writes_checksum_manifest(
    tmp_path: Path,
) -> None:
    seeded_marker = "seeded-ci-secret"
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    for filename in ("download.yaml", "stream.yaml"):
        (raw_directory / filename).write_text(
            yaml.safe_dump(_raw_cassette(seeded_marker)), encoding="utf-8"
        )

    _publish(raw_directory, publish_directory, seeded_marker)

    published_text = "".join(
        path.read_text(encoding="utf-8") for path in publish_directory.iterdir()
    )
    manifest = json.loads(
        (publish_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert seeded_marker not in published_text
    assert set(manifest["cassette_checksums"]) == {"download.yaml", "stream.yaml"}
    assert manifest["platform_revision"] == "unknown"
    assert manifest["recording_started_at"] == "2026-09-09T10:00:00Z"
    assert manifest["recording_finished_at"] == "2026-09-09T10:05:00Z"
    assert manifest["interaction_count"] == 2
    expected_bytes = sum(
        path.stat().st_size for path in publish_directory.glob("*.yaml")
    )
    assert manifest["artifact_bytes"] == expected_bytes


def test_publish_preserves_nested_cassette_paths(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    nested_directory = raw_directory / "integration_module"
    nested_directory.mkdir(parents=True)
    (nested_directory / "pilot.yaml").write_text(
        yaml.safe_dump(_raw_cassette("marker")), encoding="utf-8"
    )

    _publish(raw_directory, publish_directory, "marker")

    assert (publish_directory / "integration_module" / "pilot.yaml").is_file()
    manifest = json.loads(
        (publish_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["cassette_checksums"]) == {"integration_module/pilot.yaml"}


def test_verify_checks_exact_identity_expiry_and_checksums(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    (raw_directory / "pilot.yaml").write_text(
        yaml.safe_dump(_raw_cassette("secret")), encoding="utf-8"
    )
    _publish(raw_directory, publish_directory, "secret")

    verified = verify_cassettes(
        publish_directory=publish_directory,
        expected_sdk_sha="a" * 40,
        expected_run_id="98765",
        expected_run_attempt=1,
        now=datetime(2026, 9, 9, 12, tzinfo=timezone.utc),
    )
    assert verified.test_selection == ("tests/pilot_test.py::test_download",)

    with pytest.raises(ManifestValidationError, match="SDK SHA"):
        verify_cassettes(
            publish_directory=publish_directory,
            expected_sdk_sha="c" * 40,
            expected_run_id="98765",
            expected_run_attempt=1,
        )

    with pytest.raises(ManifestValidationError, match="expired"):
        verify_cassettes(
            publish_directory=publish_directory,
            expected_sdk_sha="a" * 40,
            expected_run_id="98765",
            expected_run_attempt=1,
            now=datetime(2026, 9, 11, tzinfo=timezone.utc),
        )

    (publish_directory / "pilot.yaml").write_text("tampered", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="checksum mismatch"):
        verify_cassettes(
            publish_directory=publish_directory,
            expected_sdk_sha="a" * 40,
            expected_run_id="98765",
            expected_run_attempt=1,
        )


def test_secret_environment_is_not_required_in_function_api(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("CASSETTE_SECRETS", "do-not-log-this")
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    (raw_directory / "pilot.yaml").write_text(
        yaml.safe_dump(_raw_cassette("do-not-log-this")), encoding="utf-8"
    )

    _publish(raw_directory, publish_directory, "do-not-log-this")

    assert "do-not-log-this" not in (publish_directory / "pilot.yaml").read_text(
        encoding="utf-8"
    )


def test_verify_rejects_unlisted_nested_yaml(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    (raw_directory / "pilot.yaml").write_text(
        yaml.safe_dump(_raw_cassette("marker")), encoding="utf-8"
    )
    _publish(raw_directory, publish_directory, "marker")
    injected_path = publish_directory / "unexpected" / "injected.yaml"
    injected_path.parent.mkdir()
    injected_path.write_text("interactions: []", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="do not match"):
        verify_cassettes(
            publish_directory=publish_directory,
            expected_sdk_sha="a" * 40,
            expected_run_id="98765",
            expected_run_attempt=1,
        )


def test_publish_filters_unrelated_git_repo_list_records(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    records: list[JsonValue] = [
        {"name": "sdk-http-recording-owned", "id": "pilot"},
        {"name": "another-team-repository", "id": "unrelated"},
    ]
    raw = _git_repo_cassette(records)
    (raw_directory / "git_repo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    _publish(raw_directory, publish_directory, "marker")

    loaded: object = yaml.safe_load(
        (publish_directory / "git_repo.yaml").read_text(encoding="utf-8")
    )
    _, published_response = _first_exchange(as_cassette(loaded))
    body_container = published_response["body"]
    headers = published_response["headers"]
    assert isinstance(body_container, dict)
    assert isinstance(headers, dict)
    body = body_container["string"]
    assert isinstance(body, str)
    assert json.loads(body) == [{"name": "sdk-http-recording-owned", "id": "pilot"}]
    assert headers["Content-Length"] == [str(len(body.encode("utf-8")))]


def test_publish_preserves_vcr_response_bytes_while_filtering(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    raw = _git_repo_cassette(
        [{"name": "sdk-http-recording-owned", "id": "pilot"}],
        body_as_bytes=True,
    )
    (raw_directory / "git_repo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    _publish(raw_directory, publish_directory, "marker")

    loaded: object = yaml.safe_load(
        (publish_directory / "git_repo.yaml").read_text(encoding="utf-8")
    )
    _, published_response = _first_exchange(as_cassette(loaded))
    body_container = published_response["body"]
    assert isinstance(body_container, dict)
    assert isinstance(body_container["string"], bytes)


def test_publish_allows_namespaced_git_repo_list_records(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    raw = _git_repo_cassette([{"name": "sdk-http-recording-owned"}])
    (raw_directory / "git_repo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    _publish(raw_directory, publish_directory, "marker")

    assert (publish_directory / "git_repo.yaml").is_file()


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [{"name": "sdk-http-recording-owned"}]},
        [{"id": "record-without-name"}],
    ],
)
def test_publish_rejects_malformed_git_repo_list_response(
    tmp_path: Path, payload: JsonValue
) -> None:
    raw_directory = tmp_path / "raw"
    publish_directory = tmp_path / "publish"
    raw_directory.mkdir()
    raw = _git_repo_cassette(payload)
    (raw_directory / "git_repo.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(UnsafeCassetteError, match="git-repo"):
        _publish(raw_directory, publish_directory, "marker")

    assert not publish_directory.exists()
