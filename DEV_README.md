# Hirundo Python SDK Development

This repo contains the source code for the Hirundo Python SDK.

## Usage

For SDK usage, see:

- Documentation site: [https://docs.hirundo.io/](https://docs.hirundo.io/)
- Example notebooks: [notebooks/](notebooks/)

Note: We currently support CPython 3.10, 3.11, 3.12, and 3.13. PyPy support may be introduced in the future.

## Development workflow

Before opening a PR, install dev dependencies and run Ruff:

```bash
ruff check
ruff format
```

### Install dev dependencies

```bash
pip install -r requirements/dev.txt
```

Optional: install and use `uv` as a faster replacement for `pip`.

### Install git hooks (optional)

```bash
pre-commit install
```

### Change dependencies

#### Update `requirements.txt` files

```bash
uv pip compile pyproject.toml
uv pip compile --extra dev -o requirements/dev.txt -c requirements.txt pyproject.toml
uv pip compile --extra pandas -o requirements/pandas.txt -c requirements.txt pyproject.toml
uv pip compile --extra polars -o requirements/polars.txt -c requirements.txt pyproject.toml
uv pip compile --extra docs -o requirements/docs.txt -c requirements.txt pyproject.toml
uv pip compile --extra transformers -o requirements/transformers.txt -c requirements.txt pyproject.toml
```

#### Sync installed packages

```bash
uv pip sync requirements/dev.txt requirements/pandas.txt requirements/polars.txt requirements/docs.txt requirements/transformers.txt
```

or

```bash
uv sync --extra dev --extra pandas --extra polars --extra docs --extra transformers
```

### Build the package

```bash
python -m build
```

### Documentation

We use Sphinx to generate documentation. To build locally, install the docs extras and run:

```bash
pip install -r requirements/docs.txt
cd docs
make html
```

Documentation releases are published via GitHub Actions when changes are merged to `main`.

### PyPI releases

New versions of `hirundo` are released via a GitHub Actions workflow that opens a PR with the new version. The package is published to PyPI when that PR is merged.
