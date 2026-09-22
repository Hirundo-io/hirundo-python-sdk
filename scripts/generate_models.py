"""Regenerate Pydantic wire models from the canonical OpenAPI snapshot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPOSITORY_ROOT / "schemas" / "openapi_snapshot.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "hirundo" / "_generated" / "wire_models.py"

app = typer.Typer(add_completion=False)


def generate(openapi_path: Path, output_path: Path) -> None:
    """Generate models directly from the canonical OpenAPI components.

    Args:
        openapi_path: Canonical OpenAPI snapshot used as generation input.
        output_path: Python module path to replace with generated models.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input",
        str(openapi_path),
        "--input-file-type",
        "openapi",
        "--openapi-scopes",
        "schemas",
        "--output",
        str(output_path),
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--disable-future-imports",
        "--target-python-version",
        "3.10",
        "--output-datetime-class",
        "datetime",
        "--use-standard-collections",
        "--use-union-operator",
        "--enum-field-as-literal",
        "one",
        "--use-subclass-enum",
        "--capitalise-enum-members",
        "--strict-nullable",
        "--strict-refs",
        "--field-constraints",
        "--deserialize-default-values",
        "enum",
        "--disable-timestamp",
        "--formatters",
        "ruff-check",
        "ruff-format",
    ]
    subprocess.run(command, check=True)  # noqa: S603


@app.command()
def main(
    schema: Annotated[
        Path,
        typer.Option(help="Canonical OpenAPI snapshot used to generate models."),
    ] = DEFAULT_SCHEMA,
    output: Annotated[
        Path,
        typer.Option(help="Python module destination for the generated models."),
    ] = DEFAULT_OUTPUT,
) -> None:
    """Parse command-line paths and regenerate the model module.

    Args:
        schema: Canonical OpenAPI snapshot used as the generation input.
        output: Python module path to replace with generated models.

    Returns:
        None.
    """
    generate(schema, output)


if __name__ == "__main__":
    app()
