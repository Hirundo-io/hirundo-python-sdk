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

Note: You need to [install](https://docs.astral.sh/uv/getting-started/installation/) and use [`uv`](https://docs.astral.sh/uv/) as a faster drop-in replacement for `pip` for our project.

Then you can install the dependencies with:

```bash
uv sync --all-groups
```

### Install `git` hooks (optional)

```bash
pre-commit install
```

### Build the package

```bash
python -m build
```

### Documentation

We use `sphinx` to generate our documentation. Note: If you want to manually create the HTML files from your documentation, you must install the `dev` dependency group.

#### Documentation releases

Documentation releases are published via GitHub Actions when changes are merged to `main`.

### PyPI releases

New versions of `hirundo` are released via a GitHub Actions workflow that opens a PR with the new version. The package is published to PyPI when that PR is merged.
