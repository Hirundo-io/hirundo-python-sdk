# Hirundo Python SDK

This repo contains the source code for the Hirundo Python SDK.

## Usage

To learn about how to use this SDK, please visit the [http://docs.hirundo.io/](documentation) or see the Google Colab examples.

Note: Currently we only support the main CPython release 3.9, 3.10, 3.11, 3.12 & 3.13. PyPy support may be introduced in the future.

## Development

When opening Pull Requests, note that the repository has GitHub Actions which run on CI/CD to lint the code and run a suite of integration tests. Please do not open a Pull Request without first installing the dev dependencies and running `ruff check` and `ruff format` on your changes.

### Install dev dependencies

```bash
pip install -r requirements/dev.txt
```

Note: You can install and use `uv` as a faster drop-in replacement for `pip`. We have it as part of our dev dependencies for this reason.

### Install `git` hooks (optional)
### Install `git` hooks (optional)

```bash
pre-commit install
```

### Check lint and apply formatting with Ruff (optional; pre-commit hooks run this automatically)

```bash
ruff check
ruff format
```

### Change packages

#### Update `requirements.txt` files

```bash
uv pip compile pyproject.toml
uv pip compile --extra dev -o requirements/dev.txt -c requirements.txt pyproject.toml
uv pip compile --extra pandas -o requirements/pandas.txt -c requirements.txt pyproject.toml
uv pip compile --extra polars -o requirements/polars.txt -c requirements.txt pyproject.toml
uv pip compile --extra docs -o requirements/docs.txt -c requirements.txt pyproject.toml
```

#### Sync installed packages

```bash
uv pip sync requirements/dev.txt requirements/polars.txt
```

### Build process

To build the package, run:
`python -m build`

### Documentation

We use `sphinx` to generate our documentation. Note: If you want to manually create the HTML files from your documentation, you must install `requirements/docs.txt` instead of/in addition to `requirements/dev.txt`.

#### Documentation releases
Documentation releases are published via GitHub Actions on merges to `main`.

### PyPI package releases

New versions of `hirundo` are released via a GitHub Actions workflow that creates a Pull Request with the version name and description, which is then published to PyPI when this Pull Request is merged.
