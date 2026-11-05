"""Fetch the deployed OpenAPI document used by CI contract pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 - Typer resolves the annotation at runtime.
from typing import Annotated
from urllib.parse import urlsplit

import typer
from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import READ_TIMEOUT

app = typer.Typer(add_completion=False)


def fetch_openapi_snapshot(output_path: Path) -> str:
    """Write a canonical schema snapshot and return its SHA-256 digest.

    Args:
        output_path: Destination for the canonical JSON document.

    Returns:
        The snapshot's SHA-256 digest, prefixed with ``sha256:``.
    """
    api_host = API_HOST
    if not api_host or urlsplit(api_host).scheme != "https":
        raise ValueError("The deployed OpenAPI pilot requires an HTTPS API host")
    response = requests.get(
        f"{api_host.rstrip('/')}/openapi.json",
        headers=get_headers(),
        timeout=READ_TIMEOUT,
    )
    raise_for_status_with_reason(response)
    schema = response.json()
    if not isinstance(schema, dict) or "openapi" not in schema or "paths" not in schema:
        raise ValueError("The target did not return an OpenAPI document")

    serialized = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()


@app.command()
def main(
    output: Annotated[
        Path,
        typer.Argument(help="Path where the canonical OpenAPI snapshot is written."),
    ],
) -> None:
    """Fetch and canonicalize the deployed OpenAPI document.

    Args:
        output: Destination for the canonical JSON document.

    Returns:
        None.
    """
    typer.echo(fetch_openapi_snapshot(output))


if __name__ == "__main__":
    app()
