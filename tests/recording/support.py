"""Recording helpers shared by the SDK's public integration tests.

The functions in this module match VCR.py's ``before_record_*`` and custom
matcher call signatures. They deliberately do not import VCR.py, so collecting
ordinary unit tests does not require the recording-only dependency.

VCR.py cannot prove that a cassette contains no secrets, and its default body
matcher compares bytes rather than JSON meaning. CI must therefore sanitize and
validate each cassette before publishing it, then install an independent network
guard while replaying it. Streaming bodies are buffered by VCR.py; the sanitizer
understands text/event-stream payloads, but it cannot reproduce packet boundaries
or timing. Binary and compressed bodies are rejected instead of being guessed at.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import ipaddress
import json
import re
import socket
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "<redacted>"
INERT_API_ORIGIN = "https://api.example.test"
SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
        "x-amz-security-token",
        "x-auth-token",
    }
)
SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "credential",
        "key",
        "sig",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
    }
)
SENSITIVE_BODY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
MATCHED_HEADER_NAMES = (
    "accept",
    "content-type",
    "hirundo-api-version",
    "x-api-version",
    "x-hirundo-api-version",
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
CassetteValue: TypeAlias = (
    JsonScalar | bytes | list["CassetteValue"] | dict[str, "CassetteValue"]
)
Cassette: TypeAlias = dict[str, CassetteValue]
ManifestValue: TypeAlias = CassetteValue | tuple[str, ...]


class VcrRequest(Protocol):
    """Request attributes used by the VCR.py hooks and matcher."""

    method: str
    uri: str
    body: CassetteValue
    headers: Mapping[str, CassetteValue]


class VcrController(Protocol):
    """VCR.py configuration operation used by pytest-recording."""

    def register_matcher(
        self, name: str, matcher: Callable[[VcrRequest, VcrRequest], None]
    ) -> None: ...


class VcrConfiguration(TypedDict):
    """pytest-recording options supplied by the recording fixture."""

    before_record_request: Callable[[VcrRequest], VcrRequest]
    before_record_response: Callable[[Mapping[str, CassetteValue]], Cassette]
    decode_compressed_response: bool
    match_on: list[str]


class UnsafeCassetteError(ValueError):
    """Raised when cassette data cannot be made safe without guessing."""


class ManifestValidationError(ValueError):
    """Raised when replay provenance is incomplete or inconsistent."""


class IdentifierCollisionError(ValueError):
    """Raised when two source identifiers would receive one replacement."""


class ReplayNetworkEscapeError(RuntimeError):
    """Raised when replay code tries to leave its recorded transport."""


def _header_values(headers: Mapping[str, CassetteValue], name: str) -> tuple[str, ...]:
    for candidate, raw_value in headers.items():
        if candidate.lower() != name:
            continue
        if isinstance(raw_value, str):
            return (raw_value.strip(),)
        if isinstance(raw_value, Sequence):
            return tuple(str(value).strip() for value in raw_value)
        return (str(raw_value).strip(),)
    return ()


def _auth_shape(headers: Mapping[str, CassetteValue]) -> tuple[bool, str | None]:
    values = _header_values(headers, "authorization")
    if not values:
        return False, None
    scheme = values[0].split(maxsplit=1)[0].lower()
    return True, scheme


def as_json_value(value: object) -> JsonValue:
    """Validate an untyped JSON decoder result.

    Args:
        value: Value returned by the JSON decoder.

    Returns:
        The recursively validated JSON value.
    """

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [as_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {str(key): as_json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def as_cassette(value: object) -> Cassette:
    """Validate an untyped YAML value as a cassette mapping.

    Args:
        value: Value returned by the YAML loader.

    Returns:
        The recursively validated cassette mapping.
    """

    parsed = _as_cassette_value(value)
    if not isinstance(parsed, dict):
        raise ValueError("cassette must contain a mapping")
    return parsed


def _as_cassette_value(value: object) -> CassetteValue:
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, list):
        return [_as_cassette_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("cassette mapping keys must be strings")
        return {str(key): _as_cassette_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported cassette value: {type(value).__name__}")


def _canonical_json_body(body: CassetteValue) -> JsonValue:
    if body in (None, b"", ""):
        return None
    if isinstance(body, str):
        body = body.encode()
    if not isinstance(body, bytes):
        raise AssertionError(f"unsupported request body type: {type(body).__name__}")
    try:
        loaded: object = json.loads(body.decode("utf-8"))
        return as_json_value(loaded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AssertionError("request body is not valid UTF-8 JSON") from error


def _json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare JSON without Python's surprising ``True == 1`` coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if not isinstance(right, dict):
            return False
        return left.keys() == right.keys() and all(
            _json_values_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        if not isinstance(right, list):
            return False
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _request_uri(request: VcrRequest) -> str:
    return request.uri


def semantic_request_matcher(left: VcrRequest, right: VcrRequest) -> None:
    """Assert that two VCR request objects have the same API meaning.

    JSON object key order and insignificant whitespace do not matter. JSON value
    types, explicit nulls, omitted keys, and array order remain significant.
    Authorization values never take part in matching, but presence and scheme do.

    Args:
        left: The incoming or recorded VCR request.
        right: The other VCR request to compare.

    Returns:
        None.
    """

    assert str(left.method).upper() == str(right.method).upper(), "method differs"
    left_url = urlsplit(_request_uri(left))
    right_url = urlsplit(_request_uri(right))
    assert left_url.scheme.lower() == right_url.scheme.lower(), "scheme differs"
    assert left_url.netloc.lower() == right_url.netloc.lower(), "authority differs"
    assert left_url.path == right_url.path, "path differs"
    assert parse_qsl(left_url.query, keep_blank_values=True) == parse_qsl(
        right_url.query, keep_blank_values=True
    ), "query differs"
    assert _json_values_equal(
        _canonical_json_body(left.body), _canonical_json_body(right.body)
    ), "JSON body differs"

    for header_name in MATCHED_HEADER_NAMES:
        assert _header_values(left.headers, header_name) == _header_values(
            right.headers, header_name
        ), f"{header_name} header differs"
    assert _auth_shape(left.headers) == _auth_shape(right.headers), (
        "authorization presence or scheme differs"
    )


class DeterministicIdentifierMapper:
    """Assign stable, relationship-preserving aliases to opaque identifiers."""

    def __init__(self) -> None:
        self._aliases_by_source: dict[tuple[str, str], str] = {}
        self._sources_by_alias: dict[str, tuple[str, str]] = {}

    def map(self, namespace: str, source: str) -> str:
        """Return a deterministic alias and reject alias collisions.

        Args:
            namespace: The identifier category used in the alias.
            source: The original identifier.

        Returns:
            The stable alias for the identifier.
        """

        source_key = (namespace, source)
        existing = self._aliases_by_source.get(source_key)
        if existing is not None:
            return existing
        digest = hashlib.sha256(f"{namespace}\0{source}".encode()).hexdigest()[:16]
        alias = f"<{namespace}-{digest}>"
        collision = self._sources_by_alias.get(alias)
        if collision is not None and collision != source_key:
            raise IdentifierCollisionError(
                f"alias {alias!r} already belongs to a different identifier"
            )
        self._aliases_by_source[source_key] = alias
        self._sources_by_alias[alias] = source_key
        return alias

    def register(self, namespace: str, source: str, alias: str) -> None:
        """Register a fixed alias, detecting conflicts in either direction.

        Args:
            namespace: The identifier category used in the alias.
            source: The original identifier.
            alias: The fixed replacement value.

        Returns:
            None.
        """

        source_key = (namespace, source)
        existing_alias = self._aliases_by_source.get(source_key)
        existing_source = self._sources_by_alias.get(alias)
        if existing_alias not in (None, alias) or existing_source not in (
            None,
            source_key,
        ):
            raise IdentifierCollisionError(f"conflicting identifier alias {alias!r}")
        self._aliases_by_source[source_key] = alias
        self._sources_by_alias[alias] = source_key


def _is_loopback(hostname: str | None) -> bool:
    if hostname in (None, "localhost"):
        return hostname == "localhost"
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _sanitize_url(url: str, *, normalize_external_host: bool = False) -> str:
    parts = urlsplit(url)
    sanitized_query = [
        (name, REDACTED if name.lower() in SENSITIVE_QUERY_NAMES else value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    scheme = parts.scheme
    netloc = parts.netloc
    if normalize_external_host and parts.hostname and not _is_loopback(parts.hostname):
        inert_parts = urlsplit(INERT_API_ORIGIN)
        scheme = inert_parts.scheme
        netloc = inert_parts.netloc
    return urlunsplit(
        (
            scheme,
            netloc,
            parts.path,
            urlencode(sanitized_query),
            parts.fragment,
        )
    )


def _sanitize_headers(
    headers: Mapping[str, CassetteValue],
) -> dict[str, CassetteValue]:
    sanitized: dict[str, CassetteValue] = {}
    for name, value in headers.items():
        if name.lower() == "authorization":
            values = value if isinstance(value, list) else [value]
            redacted_values = []
            for item in values:
                scheme = str(item).split(maxsplit=1)[0]
                redacted_values.append(f"{scheme} {REDACTED}")
            sanitized[name] = (
                redacted_values if isinstance(value, list) else redacted_values[0]
            )
        elif name.lower() in SENSITIVE_HEADER_NAMES:
            sanitized[name] = [REDACTED] if isinstance(value, list) else REDACTED
        elif name.lower() == "location" and isinstance(value, str):
            sanitized[name] = _sanitize_url(value, normalize_external_host=True)
        elif name.lower() == "location" and isinstance(value, list):
            sanitized[name] = [
                _sanitize_url(str(item), normalize_external_host=True) for item in value
            ]
        else:
            sanitized[name] = value
    return sanitized


def _sanitize_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: REDACTED
            if key.lower() in SENSITIVE_BODY_NAMES
            else _sanitize_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    return value


def _sanitize_sse(text: str) -> str:
    output_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        bare_line = line.rstrip("\r\n")
        ending = line[len(bare_line) :]
        if not bare_line.startswith("data:"):
            output_lines.append(line)
            continue
        prefix, raw_data = bare_line.split(":", maxsplit=1)
        whitespace = " " if raw_data.startswith(" ") else ""
        payload = raw_data.lstrip(" ")
        if payload == "[DONE]":
            output_lines.append(line)
            continue
        try:
            loaded: object = json.loads(payload)
            parsed = as_json_value(loaded)
        except (json.JSONDecodeError, ValueError) as error:
            raise UnsafeCassetteError("SSE data is not JSON") from error
        safe_payload = json.dumps(_sanitize_json(parsed), separators=(",", ":"))
        output_lines.append(f"{prefix}:{whitespace}{safe_payload}{ending}")
    return "".join(output_lines)


def _content_type(headers: Mapping[str, CassetteValue]) -> str:
    values = _header_values(headers, "content-type")
    return values[0].split(";", maxsplit=1)[0].lower() if values else ""


def _text_body(body: CassetteValue) -> tuple[str, bool]:
    if isinstance(body, dict):
        wrapped = "string" in body
        raw_body = body.get("string")
    else:
        wrapped = False
        raw_body = body
    if isinstance(raw_body, bytes):
        try:
            return raw_body.decode("utf-8"), wrapped
        except UnicodeDecodeError as error:
            raise UnsafeCassetteError("binary cassette body is not allowed") from error
    if isinstance(raw_body, str):
        return raw_body, wrapped
    raise UnsafeCassetteError(
        f"unsupported cassette body type: {type(raw_body).__name__}"
    )


def _sanitize_text_for_media_type(text: str, media_type: str) -> str:
    if media_type == "text/event-stream":
        return _sanitize_sse(text)
    if media_type == "application/json" or media_type.endswith("+json"):
        try:
            loaded: object = json.loads(text)
            return json.dumps(
                _sanitize_json(as_json_value(loaded)), separators=(",", ":")
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise UnsafeCassetteError("declared JSON body is invalid") from error
    if media_type.startswith("text/") or media_type in (
        "",
        "application/x-www-form-urlencoded",
    ):
        return text
    raise UnsafeCassetteError(f"unsafe cassette media type {media_type!r}")


def _sanitize_body(
    body: CassetteValue, headers: Mapping[str, CassetteValue]
) -> CassetteValue:
    encoding = ",".join(_header_values(headers, "content-encoding")).lower()
    if encoding and encoding != "identity":
        raise UnsafeCassetteError(f"compressed cassette body uses {encoding!r}")

    raw_body = body["string"] if isinstance(body, dict) and "string" in body else body
    if raw_body in (None, "", b""):
        return body
    text, wrapped = _text_body(body)
    media_type = _content_type(headers)
    sanitized_text = _sanitize_text_for_media_type(text, media_type)
    sanitized_body = (
        sanitized_text.encode("utf-8")
        if isinstance(raw_body, bytes)
        else sanitized_text
    )

    if wrapped and isinstance(body, dict):
        sanitized_wrapper = dict(body)
        sanitized_wrapper["string"] = sanitized_body
        return sanitized_wrapper
    return sanitized_body


def _sanitize_message(message: Mapping[str, CassetteValue]) -> Cassette:
    sanitized = copy.deepcopy(dict(message))
    headers = sanitized.get("headers", {})
    if not isinstance(headers, Mapping):
        raise UnsafeCassetteError("cassette headers must be a mapping")
    sanitized_headers = _sanitize_headers(headers)
    sanitized["headers"] = sanitized_headers
    uri = sanitized.get("uri")
    if isinstance(uri, str):
        sanitized["uri"] = _sanitize_url(uri, normalize_external_host=True)
    url = sanitized.get("url")
    if isinstance(url, str):
        sanitized["url"] = _sanitize_url(url, normalize_external_host=True)
    if "body" in sanitized:
        sanitized["body"] = _sanitize_body(sanitized["body"], sanitized_headers)
    return sanitized


def _json_secret_scan_default(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"unsupported secret-scan value: {type(value).__name__}")


def sanitize_cassette(
    cassette: Mapping[str, CassetteValue], *, seeded_secrets: Sequence[str]
) -> Cassette:
    """Return a publishable cassette or fail if any seeded secret survives.

    Args:
        cassette: The loaded VCR YAML document.
        seeded_secrets: Secret markers that must not survive sanitization.

    Returns:
        A sanitized copy of the cassette.
    """

    sanitized = copy.deepcopy(dict(cassette))
    interactions = sanitized.get("interactions")
    if not isinstance(interactions, list):
        raise UnsafeCassetteError("cassette interactions must be a list")
    sanitized_interactions: list[CassetteValue] = []
    for interaction in interactions:
        if not isinstance(interaction, dict):
            raise UnsafeCassetteError("cassette interaction must be a mapping")
        request = interaction.get("request")
        response = interaction.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise UnsafeCassetteError(
                "interaction request and response must be mappings"
            )
        sanitized_interactions.append(
            {
                "request": _sanitize_message(request),
                "response": _sanitize_message(response),
            }
        )
    sanitized["interactions"] = sanitized_interactions
    serialized = json.dumps(
        sanitized,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_secret_scan_default,
    )
    for secret in seeded_secrets:
        if not secret:
            raise UnsafeCassetteError("seeded secrets must not be empty")
        encodings = {
            secret,
            urlencode({"value": secret}).partition("=")[2],
            base64.b64encode(secret.encode()).decode(),
        }
        if any(encoded_secret in serialized for encoded_secret in encodings):
            raise UnsafeCassetteError("a seeded secret survived cassette sanitization")
    return sanitized


def before_record_request(request: VcrRequest) -> VcrRequest:
    """Strip request credentials before VCR.py serialization.

    Args:
        request: The VCR.py request being serialized.

    Returns:
        The sanitized request object.
    """

    request.uri = _sanitize_url(request.uri, normalize_external_host=True)
    request.headers = _sanitize_headers(request.headers)
    request.body = _sanitize_body(request.body, request.headers)
    return request


def before_record_response(response: Mapping[str, CassetteValue]) -> Cassette:
    """Strip response secrets before VCR.py serialization.

    Args:
        response: The VCR.py response mapping being serialized.

    Returns:
        A sanitized response mapping.
    """

    return _sanitize_message(response)


@dataclass(frozen=True)
class RecordingManifest:
    """Provenance that binds replay artifacts to one CI recording job."""

    sdk_sha: str
    run_id: str
    run_attempt: int
    environment: str
    recording_started_at: str
    recording_finished_at: str
    expires_at: str
    schema_digest: str
    platform_revision: str
    tool_versions: Mapping[str, str]
    test_selection: tuple[str, ...]
    cassette_format: str
    interaction_count: int
    artifact_bytes: int
    cassette_checksums: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, ManifestValue]) -> RecordingManifest:
        """Build and validate a recording manifest.

        Args:
            value: The manifest fields loaded from JSON.

        Returns:
            The validated recording manifest.
        """

        expected_fields = {
            "sdk_sha",
            "run_id",
            "run_attempt",
            "environment",
            "recording_started_at",
            "recording_finished_at",
            "expires_at",
            "schema_digest",
            "platform_revision",
            "tool_versions",
            "test_selection",
            "cassette_format",
            "interaction_count",
            "artifact_bytes",
            "cassette_checksums",
        }
        if set(value) != expected_fields:
            raise ManifestValidationError("manifest fields are missing or unexpected")
        manifest = cls(
            sdk_sha=_manifest_string(value, "sdk_sha"),
            run_id=_manifest_string(value, "run_id"),
            run_attempt=_manifest_integer(value, "run_attempt"),
            environment=_manifest_string(value, "environment"),
            recording_started_at=_manifest_string(value, "recording_started_at"),
            recording_finished_at=_manifest_string(value, "recording_finished_at"),
            expires_at=_manifest_string(value, "expires_at"),
            schema_digest=_manifest_string(value, "schema_digest"),
            platform_revision=_manifest_string(value, "platform_revision"),
            tool_versions=_manifest_string_mapping(value, "tool_versions"),
            test_selection=_manifest_string_tuple(value, "test_selection"),
            cassette_format=_manifest_string(value, "cassette_format"),
            interaction_count=_manifest_integer(value, "interaction_count"),
            artifact_bytes=_manifest_integer(value, "artifact_bytes"),
            cassette_checksums=_manifest_string_mapping(value, "cassette_checksums"),
        )
        manifest.validate()
        return manifest

    def validate(self, cassette_root: Path | None = None) -> None:
        """Validate manifest fields and optional cassette files.

        Args:
            cassette_root: Directory containing the published cassettes, if available.

        Returns:
            None.
        """

        self._validate_provenance()
        self._validate_artifacts(cassette_root)

    def _validate_provenance(self) -> None:
        if not SHA_PATTERN.fullmatch(self.sdk_sha):
            raise ManifestValidationError("sdk_sha must be a 40 or 64 digit commit SHA")
        if not self.run_id.strip() or self.run_attempt < 1:
            raise ManifestValidationError("run ID and positive attempt are required")
        if not self.environment.strip():
            raise ManifestValidationError("recording environment is required")
        self._validate_times()
        if not SHA256_PATTERN.fullmatch(self.schema_digest):
            raise ManifestValidationError("schema_digest must be a sha256 digest")
        if self.platform_revision != "unknown" and not SHA_PATTERN.fullmatch(
            self.platform_revision
        ):
            raise ManifestValidationError(
                "platform_revision must be a commit SHA or explicit 'unknown'"
            )
        if not self.tool_versions or any(
            not name.strip() or not version.strip()
            for name, version in self.tool_versions.items()
        ):
            raise ManifestValidationError("tool_versions must contain named versions")
        if not self.test_selection or any(
            not test.strip() for test in self.test_selection
        ):
            raise ManifestValidationError("test_selection must contain pytest node IDs")
        if self.cassette_format != "vcrpy-yaml-v1":
            raise ManifestValidationError("unsupported cassette format")
        self._validate_measurements()

    def _validate_measurements(self) -> None:
        if self.interaction_count < 1:
            raise ManifestValidationError("interaction_count must be positive")
        if self.artifact_bytes < 1:
            raise ManifestValidationError("artifact_bytes must be positive")

    def _validate_times(self) -> None:
        started_at = _parse_utc_timestamp(
            self.recording_started_at, "recording_started_at"
        )
        finished_at = _parse_utc_timestamp(
            self.recording_finished_at, "recording_finished_at"
        )
        expires_at = _parse_utc_timestamp(self.expires_at, "expires_at")
        if finished_at < started_at:
            raise ManifestValidationError("recording finish must not precede its start")
        if expires_at <= finished_at:
            raise ManifestValidationError("expires_at must follow recording finish")

    def _validate_artifacts(self, cassette_root: Path | None) -> None:
        if not self.cassette_checksums:
            raise ManifestValidationError("cassette_checksums must not be empty")
        if cassette_root is not None:
            actual_names = {
                path.relative_to(cassette_root).as_posix()
                for path in cassette_root.rglob("*.yaml")
            }
            if actual_names != set(self.cassette_checksums):
                raise ManifestValidationError(
                    "published YAML files do not match the manifest"
                )
        for filename, expected_digest in self.cassette_checksums.items():
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ManifestValidationError(
                    "cassette paths must stay under their root"
                )
            if not SHA256_PATTERN.fullmatch(expected_digest):
                raise ManifestValidationError(f"invalid checksum for {filename}")
            if cassette_root is not None:
                try:
                    cassette_bytes = (cassette_root / filename).read_bytes()
                except OSError as error:
                    raise ManifestValidationError(
                        f"cannot read cassette {filename}"
                    ) from error
                actual_digest = hashlib.sha256(cassette_bytes).hexdigest()
                if expected_digest != f"sha256:{actual_digest}":
                    raise ManifestValidationError(f"checksum mismatch for {filename}")

    def verify_replay(
        self,
        *,
        cassette_root: Path,
        expected_sdk_sha: str,
        expected_run_id: str,
        expected_run_attempt: int,
        now: datetime | None = None,
    ) -> None:
        """Validate the files and exact CI identity before enabling replay.

        Args:
            cassette_root: Directory containing the published cassettes.
            expected_sdk_sha: Exact SDK commit expected by the replay job.
            expected_run_id: Exact CI run identifier expected by the replay job.
            expected_run_attempt: Exact CI attempt expected by the replay job.
            now: UTC time used for expiry validation, or the current time when omitted.

        Returns:
            None.
        """

        self.validate(cassette_root)
        if self.sdk_sha != expected_sdk_sha:
            raise ManifestValidationError("SDK SHA does not match replay job")
        if self.run_id != expected_run_id:
            raise ManifestValidationError("run ID does not match replay job")
        if self.run_attempt != expected_run_attempt:
            raise ManifestValidationError("run attempt does not match replay job")
        current_time = now or datetime.now(timezone.utc)
        if current_time > _parse_utc_timestamp(self.expires_at, "expires_at"):
            raise ManifestValidationError("recording artifacts have expired")


def _parse_utc_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ManifestValidationError(f"{field_name} must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ManifestValidationError(f"{field_name} must use UTC")
    return parsed


def _manifest_string(value: Mapping[str, ManifestValue], field_name: str) -> str:
    field_value = value[field_name]
    if not isinstance(field_value, str):
        raise ManifestValidationError(f"{field_name} must be text")
    return field_value


def _manifest_integer(value: Mapping[str, ManifestValue], field_name: str) -> int:
    field_value = value[field_name]
    if not isinstance(field_value, int) or isinstance(field_value, bool):
        raise ManifestValidationError(f"{field_name} must be an integer")
    return field_value


def _manifest_string_mapping(
    value: Mapping[str, ManifestValue], field_name: str
) -> dict[str, str]:
    field_value = value[field_name]
    if not isinstance(field_value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in field_value.items()
    ):
        raise ManifestValidationError(f"{field_name} must map text to text")
    return {str(key): str(item) for key, item in field_value.items()}


def _manifest_string_tuple(
    value: Mapping[str, ManifestValue], field_name: str
) -> tuple[str, ...]:
    field_value = value[field_name]
    if not isinstance(field_value, list | tuple) or not all(
        isinstance(item, str) for item in field_value
    ):
        raise ManifestValidationError(f"{field_name} must be a list of text")
    return tuple(str(item) for item in field_value)


@contextlib.contextmanager
def block_replay_network() -> Iterator[None]:
    """Block socket creation independently of the cassette replay library.

    Args:
        None.

    Returns:
        A context manager iterator that yields no value.
    """

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise ReplayNetworkEscapeError(
            "network access is forbidden during cassette replay"
        )

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
    ):
        yield


def vcr_config() -> VcrConfiguration:
    """Return strict pytest-recording configuration for SDK cassettes.

    Args:
        None.

    Returns:
        The pytest-recording configuration dictionary.
    """

    return {
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
        "decode_compressed_response": False,
        "match_on": ["semantic_request"],
    }


def configure_vcr(vcr: VcrController) -> VcrConfiguration:
    """Register the semantic matcher and return the corresponding config.

    Args:
        vcr: The pytest-recording VCR instance to configure.

    Returns:
        The matching pytest-recording configuration dictionary.
    """

    vcr.register_matcher("semantic_request", semantic_request_matcher)
    return vcr_config()
