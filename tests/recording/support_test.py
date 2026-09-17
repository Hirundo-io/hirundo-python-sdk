from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

import httpx
import pytest
import requests
from requests.adapters import HTTPAdapter
from tests.recording.support import (
    Cassette,
    CassetteValue,
    DeterministicIdentifierMapper,
    IdentifierCollisionError,
    ManifestValidationError,
    ManifestValue,
    RecordingManifest,
    ReplayNetworkEscapeError,
    UnsafeCassetteError,
    VcrRequest,
    before_record_request,
    block_replay_network,
    configure_vcr,
    sanitize_cassette,
    semantic_request_matcher,
    validate_cassette_secrets,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


@dataclass
class StubVcrRequest:
    method: str
    uri: str
    body: CassetteValue
    headers: Mapping[str, CassetteValue]


class RequestChange(TypedDict, total=False):
    body: str
    headers: Mapping[str, CassetteValue]
    uri: str


class FakeVcr:
    def __init__(self) -> None:
        self.semantic_request: Callable[[VcrRequest, VcrRequest], None] | None = None

    def register_matcher(
        self,
        name: str,
        matcher: Callable[[VcrRequest, VcrRequest], None],
    ) -> None:
        assert name == "semantic_request"
        self.semantic_request = matcher


def request(
    *,
    body: str = '{"enabled":true,"labels":["a","b"],"value":null}',
    headers: Mapping[str, CassetteValue] | None = None,
    uri: str = "https://api.example.test/jobs?limit=2&active=true",
) -> StubVcrRequest:
    return StubVcrRequest(
        method="POST",
        uri=uri,
        body=body.encode(),
        headers=headers
        or {
            "Accept": ["application/json"],
            "Content-Type": ["application/json"],
            "X-API-Version": ["2026-09-01"],
            "Authorization": ["Bearer first-secret"],
        },
    )


def test_semantic_matcher_ignores_json_format_and_credential_value() -> None:
    left = request()
    right = request(
        body='{"value": null, "labels": ["a", "b"], "enabled": true}',
        headers={**left.headers, "Authorization": ["Bearer second-secret"]},
    )

    semantic_request_matcher(left, right)


def test_semantic_matcher_rejects_actual_hirundo_api_version_change() -> None:
    left = request(
        headers={
            **request().headers,
            "HIRUNDO-API-VERSION": ["2026-09-01"],
        }
    )
    right = request(
        headers={
            **request().headers,
            "HIRUNDO-API-VERSION": ["2026-09-02"],
        }
    )

    with pytest.raises(AssertionError, match="hirundo-api-version"):
        semantic_request_matcher(left, right)


@pytest.mark.parametrize(
    "change",
    [
        {"body": '{"enabled":1,"labels":["a","b"],"value":null}'},
        {"body": '{"enabled":true,"labels":["b","a"],"value":null}'},
        {"body": '{"enabled":true,"labels":["a","b"]}'},
        {"uri": "https://api.example.test/jobs?active=true&limit=2"},
        {"headers": {"Authorization": ["Basic dXNlcjpwYXNz"]}},
    ],
)
def test_semantic_matcher_rejects_meaningful_differences(
    change: RequestChange,
) -> None:
    with pytest.raises(AssertionError):
        semantic_request_matcher(request(), request(**change))


def test_identifier_mapping_is_stable_and_preserves_relationships() -> None:
    mapper = DeterministicIdentifierMapper()

    first = mapper.map("job", "real-job-id")

    assert mapper.map("job", "real-job-id") == first
    assert mapper.map("job", "another-job-id") != first
    assert DeterministicIdentifierMapper().map("job", "real-job-id") == first


def test_identifier_mapping_detects_collisions() -> None:
    mapper = DeterministicIdentifierMapper()
    mapper.register("job", "first", "<job-fixed>")

    with pytest.raises(IdentifierCollisionError):
        mapper.register("job", "second", "<job-fixed>")


def cassette() -> Cassette:
    return {
        "version": 1,
        "interactions": [
            {
                "request": {
                    "method": "POST",
                    "uri": "https://api.example.test/jobs?token=seed-secret&limit=2",
                    "headers": {
                        "Authorization": ["Bearer seed-secret"],
                        "Cookie": ["session=seed-secret"],
                        "Content-Type": ["application/json"],
                    },
                    "body": {
                        "string": json.dumps(
                            {"token": "seed-secret", "nested": {"keep": None}}
                        )
                    },
                },
                "response": {
                    "status": {"code": 302, "message": "Found"},
                    "headers": {
                        "Location": [
                            "https://download.example.test/file?signature=seed-secret"
                        ],
                        "Set-Cookie": ["session=seed-secret"],
                        "Content-Type": ["text/event-stream"],
                    },
                    "body": {
                        "string": (
                            'event: progress\ndata: {"token":"seed-secret",'
                            '"nested":{"password":"seed-secret","keep":7}}\n\n'
                        )
                    },
                },
            }
        ],
    }


def first_exchange(recorded: Cassette) -> tuple[Cassette, Cassette]:
    interactions = recorded["interactions"]
    assert isinstance(interactions, list)
    interaction = interactions[0]
    assert isinstance(interaction, dict)
    request_message = interaction["request"]
    response_message = interaction["response"]
    assert isinstance(request_message, dict)
    assert isinstance(response_message, dict)
    return request_message, response_message


def body_container(message: Cassette) -> Cassette:
    body = message["body"]
    assert isinstance(body, dict)
    return body


def message_headers(message: Cassette) -> Cassette:
    headers = message["headers"]
    assert isinstance(headers, dict)
    return headers


def test_sanitizer_covers_nested_json_sse_query_cookie_and_redirect() -> None:
    sanitized = sanitize_cassette(cassette(), seeded_secrets=["seed-secret"])
    serialized = json.dumps(sanitized)
    request_message, response_message = first_exchange(sanitized)
    request_body_text = body_container(request_message)["string"]
    response_body = body_container(response_message)["string"]
    assert isinstance(request_body_text, str)
    assert isinstance(response_body, str)
    request_body: object = json.loads(request_body_text)
    assert isinstance(request_body, dict)
    nested = request_body["nested"]
    assert isinstance(nested, dict)

    assert "seed-secret" not in serialized
    request_uri = request_message["uri"]
    assert isinstance(request_uri, str)
    assert "%3Credacted%3E" in request_uri
    assert message_headers(request_message)["Authorization"] == ["Bearer <redacted>"]
    assert nested["keep"] is None
    assert '"keep":7' in response_body
    assert "[DONE]" not in serialized


def test_sanitizer_redacts_nested_credentials_and_sanitizes_json_urls() -> None:
    nested_private_key = "nested-private-key"
    nested_credential = "nested-credential"
    signed_url_marker = "signed-url-marker"
    recorded = cassette()
    request_message, response_message = first_exchange(recorded)
    request_headers = message_headers(request_message)
    request_headers["Content-Length"] = ["999"]
    request_message["body"] = {
        "string": json.dumps(
            {
                "connection": {
                    "private_key": nested_private_key,
                    "credential": nested_credential,
                },
                "download_url": (
                    "https://storage.internal.example/download?"
                    f"X-Amz-Signature={signed_url_marker}&part=1"
                ),
            }
        )
    }
    response_headers = message_headers(response_message)
    response_headers["content-length"] = "999"
    response_headers["Content-Length"] = 999
    response_headers["Content-Type"] = ["application/json"]
    response_message["body"] = {
        "string": json.dumps(
            {
                "nested": {"privateKey": nested_private_key},
                "callback": (
                    f"https://callback.internal.example/?signature={signed_url_marker}"
                ),
            }
        )
    }

    sanitized = sanitize_cassette(
        recorded,
        seeded_secrets=[nested_private_key, nested_credential, signed_url_marker],
    )

    sanitized_request, sanitized_response = first_exchange(sanitized)
    request_body = json.loads(str(body_container(sanitized_request)["string"]))
    response_body = json.loads(str(body_container(sanitized_response)["string"]))
    assert request_body["connection"] == {
        "private_key": "<redacted>",
        "credential": "<redacted>",
    }
    assert request_body["download_url"] == (
        "https://api.example.test/download?X-Amz-Signature=%3Credacted%3E&part=1"
    )
    assert response_body["nested"] == {"privateKey": "<redacted>"}
    assert response_body["callback"] == (
        "https://api.example.test/?signature=%3Credacted%3E"
    )
    sanitized_request_headers = message_headers(sanitized_request)
    sanitized_response_headers = message_headers(sanitized_response)
    assert sanitized_request_headers["Content-Length"] == [
        str(len(str(body_container(sanitized_request)["string"]).encode("utf-8")))
    ]
    assert sanitized_response_headers["content-length"] == str(
        len(str(body_container(sanitized_response)["string"]).encode("utf-8"))
    )
    assert "Content-Length" not in sanitized_response_headers


def test_sanitizer_redacts_storage_provider_credentials() -> None:
    recorded = cassette()
    request_message, _ = first_exchange(recorded)
    request_message["body"] = {
        "string": json.dumps(
            {
                "gcp": {"credentials_json": {"private_key": "gcp-private-key"}},
                "s3": {
                    "access_key_id": "aws-access-key",
                    "secret_access_key": "aws-secret-key",
                },
                "git": {"password": "huggingface-token"},
            }
        )
    }

    sanitized = sanitize_cassette(
        recorded,
        seeded_secrets=[
            "gcp-private-key",
            "aws-access-key",
            "aws-secret-key",
            "huggingface-token",
        ],
    )

    sanitized_request, _ = first_exchange(sanitized)
    request_body = json.loads(str(body_container(sanitized_request)["string"]))
    assert request_body == {
        "gcp": {"credentials_json": "<redacted>"},
        "s3": {
            "access_key_id": "<redacted>",
            "secret_access_key": "<redacted>",
        },
        "git": {"password": "<redacted>"},
    }


def test_before_record_request_recalculates_case_insensitive_content_length() -> None:
    recorded_request = request(
        body='{"token":"seed-secret"}',
        headers={
            "Content-Type": ["application/json"],
            "CONTENT-LENGTH": ["999"],
        },
    )

    sanitized_request = before_record_request(recorded_request)

    assert sanitized_request.headers["CONTENT-LENGTH"] == [
        str(len(b'{"token":"<redacted>"}'))
    ]


def test_secret_scan_extracts_json_values_without_treating_syntax_as_secret() -> None:
    private_key = "nested-private-key-material"
    credentials_document = json.dumps(
        {"type": "service_account", "private_key": private_key}, indent=2
    )

    validate_cassette_secrets(
        {"interactions": [], "safe": "{}"},
        seeded_secrets=[credentials_document],
    )
    with pytest.raises(UnsafeCassetteError, match="seeded secret"):
        validate_cassette_secrets(
            {"interactions": [], "leaked": private_key},
            seeded_secrets=[credentials_document],
        )


def test_sanitizer_preserves_binary_container_for_utf8_response_bytes() -> None:
    recorded = cassette()
    _, response_message = first_exchange(recorded)
    response_body = body_container(response_message)
    response_body_text = response_body["string"]
    assert isinstance(response_body_text, str)
    response_body["string"] = response_body_text.encode("utf-8")

    sanitized = sanitize_cassette(recorded, seeded_secrets=["seed-secret"])

    _, sanitized_response = first_exchange(sanitized)
    assert isinstance(body_container(sanitized_response)["string"], bytes)


def test_sanitizer_replaces_external_origins_but_preserves_loopback() -> None:
    external = cassette()
    external_request, _ = first_exchange(external)
    external_request["uri"] = "https://secret-api.internal.example/jobs?limit=2"
    sanitized = sanitize_cassette(external, seeded_secrets=["seed-secret"])
    sanitized_request, sanitized_response = first_exchange(sanitized)

    assert sanitized_request["uri"] == "https://api.example.test/jobs?limit=2"
    assert message_headers(sanitized_response)["Location"] == [
        "https://api.example.test/file?signature=%3Credacted%3E"
    ]

    loopback = cassette()
    loopback_request, _ = first_exchange(loopback)
    loopback_request["uri"] = "http://127.0.0.1:8123/jobs"
    sanitized_loopback = sanitize_cassette(loopback, seeded_secrets=["seed-secret"])
    sanitized_loopback_request, _ = first_exchange(sanitized_loopback)
    assert sanitized_loopback_request["uri"] == "http://127.0.0.1:8123/jobs"


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"Content-Encoding": ["gzip"]}, {"string": "opaque"}),
        ({"Content-Type": ["application/octet-stream"]}, {"string": "opaque"}),
        ({"Content-Type": ["application/json"]}, {"string": b"\xff\xfe"}),
    ],
)
def test_sanitizer_fails_closed_for_unsafe_bodies(
    headers: Cassette, body: Cassette
) -> None:
    unsafe = cassette()
    _, unsafe_response = first_exchange(unsafe)
    unsafe_response.update(headers=headers, body=body)

    with pytest.raises(UnsafeCassetteError):
        sanitize_cassette(unsafe, seeded_secrets=["seed-secret"])


def manifest(checksum: str) -> RecordingManifest:
    return RecordingManifest(
        sdk_sha="a" * 40,
        run_id="123456",
        run_attempt=2,
        environment="designated-test",
        recording_started_at="2026-09-09T10:00:00Z",
        recording_finished_at="2026-09-09T10:05:00Z",
        expires_at="2026-09-10T10:00:00Z",
        schema_digest="sha256:" + "b" * 64,
        platform_revision="unknown",
        tool_versions={"vcrpy": "7.0.0", "pytest-recording": "0.13.4"},
        test_selection=("tests/pilot_test.py::test_stream",),
        cassette_format="vcrpy-yaml-v1",
        interaction_count=1,
        artifact_bytes=100,
        cassette_checksums={"pilot.yaml": checksum},
    )


def manifest_values(checksum: str) -> dict[str, ManifestValue]:
    value = manifest(checksum)
    return {
        "sdk_sha": value.sdk_sha,
        "run_id": value.run_id,
        "run_attempt": value.run_attempt,
        "environment": value.environment,
        "recording_started_at": value.recording_started_at,
        "recording_finished_at": value.recording_finished_at,
        "expires_at": value.expires_at,
        "schema_digest": value.schema_digest,
        "platform_revision": value.platform_revision,
        "tool_versions": dict(value.tool_versions),
        "test_selection": value.test_selection,
        "cassette_format": value.cassette_format,
        "interaction_count": value.interaction_count,
        "artifact_bytes": value.artifact_bytes,
        "cassette_checksums": dict(value.cassette_checksums),
    }


def test_manifest_binds_provenance_and_artifact_checksum(tmp_path: Path) -> None:
    cassette_path = tmp_path / "pilot.yaml"
    cassette_path.write_text("safe cassette", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(cassette_path.read_bytes()).hexdigest()

    manifest(checksum).validate(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sdk_sha", "main"),
        ("run_attempt", 0),
        ("recording_started_at", "2026-09-09T10:00:00"),
        ("schema_digest", "b" * 64),
        ("platform_revision", "latest"),
        ("tool_versions", {}),
        ("interaction_count", 0),
        ("artifact_bytes", 0),
    ],
)
def test_manifest_rejects_incomplete_or_ambiguous_provenance(
    field: str, value: ManifestValue
) -> None:
    values = manifest_values("sha256:" + "c" * 64)
    values[field] = value

    with pytest.raises(ManifestValidationError):
        RecordingManifest.from_mapping(values)


def test_manifest_detects_cassette_tampering(tmp_path: Path) -> None:
    (tmp_path / "pilot.yaml").write_text("changed", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="checksum mismatch"):
        manifest("sha256:" + "c" * 64).validate(tmp_path)


def test_manifest_replay_requires_exact_ci_identity_and_unexpired_files(
    tmp_path: Path,
) -> None:
    cassette_path = tmp_path / "pilot.yaml"
    cassette_path.write_text("safe cassette", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(cassette_path.read_bytes()).hexdigest()

    with pytest.raises(ManifestValidationError, match="run attempt"):
        manifest(checksum).verify_replay(
            cassette_root=tmp_path,
            expected_sdk_sha="a" * 40,
            expected_run_id="123456",
            expected_run_attempt=3,
            expected_schema_digest="sha256:" + "b" * 64,
            expected_test_selection=("tests/pilot_test.py::test_stream",),
        )


def test_manifest_replay_rejects_expiry_equality(tmp_path: Path) -> None:
    cassette_path = tmp_path / "pilot.yaml"
    cassette_path.write_text("safe cassette", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(cassette_path.read_bytes()).hexdigest()

    with pytest.raises(ManifestValidationError, match="expired"):
        manifest(checksum).verify_replay(
            cassette_root=tmp_path,
            expected_sdk_sha="a" * 40,
            expected_run_id="123456",
            expected_run_attempt=2,
            expected_schema_digest="sha256:" + "b" * 64,
            expected_test_selection=("tests/pilot_test.py::test_stream",),
            now=datetime(2026, 9, 10, 10, tzinfo=timezone.utc),
        )


def test_configure_vcr_registers_semantic_matcher_and_strict_hooks() -> None:
    fake_vcr = FakeVcr()

    config = configure_vcr(fake_vcr)

    assert fake_vcr.semantic_request is semantic_request_matcher
    assert config["match_on"] == ["semantic_request"]
    assert "record_mode" not in config


@pytest.mark.parametrize("client", ["requests", "httpx"])
def test_replay_network_guard_blocks_unrecorded_requests(client: str) -> None:
    with (
        block_replay_network(),
        pytest.raises(
            (ReplayNetworkEscapeError, requests.RequestException, httpx.HTTPError)
        ) as captured,
    ):
        if client == "requests":
            requests.get("http://127.0.0.1:9/unrecorded", timeout=0.1)
        else:
            httpx.get("http://127.0.0.1:9/unrecorded", timeout=0.1)

    assert "forbidden during cassette replay" in _exception_chain(captured.value)


def test_replay_network_guard_blocks_httpx_redirect_escape() -> None:
    network_transport = httpx.HTTPTransport()

    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.host == "recorded.example.test":
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1:9/unrecorded"},
                request=request,
            )
        return network_transport.handle_request(request)

    with (
        httpx.Client(
            transport=httpx.MockTransport(redirect), follow_redirects=True
        ) as client,
        block_replay_network(),
        pytest.raises(ReplayNetworkEscapeError) as captured,
    ):
        client.get("https://recorded.example.test/start")

    assert "forbidden during cassette replay" in _exception_chain(captured.value)


def test_replay_network_guard_blocks_requests_redirect_escape() -> None:
    network_adapter = HTTPAdapter()
    redirect_adapter = HTTPAdapter()

    def redirect_once(
        request: requests.PreparedRequest,
        *,
        stream: bool = False,
        timeout: float | tuple[float, float] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        request_url = request.url
        assert request_url is not None
        if request_url.startswith("https://recorded.example.test/"):
            response = requests.Response()
            response.status_code = 302
            response.url = request_url
            response.request = request
            response.headers["Location"] = "http://127.0.0.1:9/unrecorded"
            return response
        return network_adapter.send(request)

    with (
        requests.Session() as session,
        block_replay_network(),
        patch.object(redirect_adapter, "send", side_effect=redirect_once),
        pytest.raises(ReplayNetworkEscapeError) as captured,
    ):
        session.mount("http://", redirect_adapter)
        session.mount("https://", redirect_adapter)
        session.get("https://recorded.example.test/start", timeout=0.1)

    assert "forbidden during cassette replay" in _exception_chain(captured.value)


def _exception_chain(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(messages)
