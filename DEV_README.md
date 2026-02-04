# Hirundo Python SDK

This repo contains the source code for the Hirundo Python SDK.

## Usage

To learn about how to use this SDK, please visit the [http://docs.hirundo.io/](documentation) or see the Google Colab examples.

Note: Currently we only support the main CPython release 3.10, 3.11, 3.12 & 3.13. PyPy support may be introduced in the future.

## Development

When opening Pull Requests, note that the repository has GitHub Actions which run on CI/CD to lint the code and run a suite of integration tests. Please do not open a Pull Request without first installing the dev dependencies and running `ruff check` and `ruff format` on your changes.

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

### Check lint and apply formatting with Ruff (optional; pre-commit hooks run this automatically)

```bash
ruff check
ruff format
```

### Build process

To build the package, run:
`python -m build`

### Documentation

We use `sphinx` to generate our documentation. Note: If you want to manually create the HTML files from your documentation, you must install the `dev` dependency group.

#### Documentation releases
Documentation releases are published via GitHub Actions on merges to `main`.

### PyPI package releases

New versions of `hirundo` are released via a GitHub Actions workflow that creates a Pull Request with the version name and description, which is then published to PyPI when this Pull Request is merged.
