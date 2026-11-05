"""Run the bounded, read-only Schemathesis diagnostic."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SAFE_PATHS = (
    "/llm-behavior-eval/run/info/{hir_run_id}",
    "/unlearning-llm/llm/{llm_model_id}",
)
MAX_EXAMPLES = 3
MAX_RUN_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 5
PLATFORM_REVISION = "unknown"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "schemathesis_pilot.toml"
)


def build_command(openapi_url: str, api_host: str, version: str) -> list[str]:
    """Build the reproducible CLI invocation for the safe GET allowlist.

    Args:
        openapi_url: URL or local path for the OpenAPI document under test.
        api_host: Base URL that receives generated API requests.
        version: Public API version header value.

    Returns:
        Subprocess arguments for the bounded Schemathesis run.
    """
    command = [
        sys.executable,
        "-m",
        "schemathesis",
        "--config-file",
        str(CONFIG_PATH),
        "run",
        openapi_url,
        "--url",
        api_host,
        "--include-method",
        "GET",
        "--phases",
        "examples,coverage,fuzzing",
        "--mode",
        "all",
        "--checks",
        "not_a_server_error,status_code_conformance,response_schema_conformance,negative_data_rejection",
        "--max-examples",
        str(MAX_EXAMPLES),
        "--max-time",
        str(MAX_RUN_SECONDS),
        "--request-timeout",
        str(REQUEST_TIMEOUT_SECONDS),
        "--request-retries",
        "0",
        "--max-failures",
        "3",
        "--workers",
        "1",
        "--seed",
        "133",
        "--generation-deterministic",
        "--generation-database",
        "none",
        "--header",
        f"HIRUNDO-API-VERSION:{version}",
        "--output-sanitize",
        "true",
    ]
    for path in SAFE_PATHS:
        command.extend(("--include-path", path))
    return command


def main() -> int:
    """Run only in the recorder job, and skip cleanly without credentials.

    Args:
        None.

    Returns:
        The Schemathesis process exit code, or zero when the pilot is skipped.
    """
    if os.environ.get("HTTP_CONTRACT_PILOT") != "1":
        print("Schemathesis pilot skipped: it runs once in the recorder job only.")
        return 0
    required_environment = {
        "SCHEMATHESIS_OPENAPI_URL": os.environ.get("SCHEMATHESIS_OPENAPI_URL"),
        "HIRUNDO_API_HOST": os.environ.get("HIRUNDO_API_HOST"),
        "HIRUNDO_API_TOKEN": os.environ.get("HIRUNDO_API_TOKEN"),
        "HIRUNDO_API_VERSION": os.environ.get("HIRUNDO_API_VERSION"),
    }
    missing = [name for name, value in required_environment.items() if not value]
    if missing:
        print(f"Schemathesis pilot skipped: missing {', '.join(missing)}.")
        return 0
    print(
        "Schemathesis diagnostic: platform revision is unknown; "
        "results do not identify or validate a particular platform commit."
    )
    print(
        f"Bounds: {len(SAFE_PATHS)} GET operations, {MAX_EXAMPLES} examples per "
        f"operation, {MAX_RUN_SECONDS}s total, {REQUEST_TIMEOUT_SECONDS}s per request."
    )
    command = build_command(
        required_environment["SCHEMATHESIS_OPENAPI_URL"] or "",
        required_environment["HIRUNDO_API_HOST"] or "",
        required_environment["HIRUNDO_API_VERSION"] or "",
    )
    started_at = time.monotonic()
    # The executable is the current interpreter and all options are separate argv.
    return_code = subprocess.run(command, check=False).returncode  # noqa: S603
    elapsed_seconds = time.monotonic() - started_at
    print(
        f"Schemathesis diagnostic finished in {elapsed_seconds:.2f}s. "
        "The CLI report records request cost; no deployed revision is available."
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
