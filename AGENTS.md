# Repository Guidelines

## Project Structure & Module Organization

- `hirundo/` holds the SDK source (CLI entry point is `hirundo.cli:app`).
- `tests/` contains pytest-based test coverage.
- `docs/` and `source/` contain Sphinx documentation assets.
- `notebooks/` and `on_prem_test_notebook.ipynb` provide example workflows.
- `requirements/` stores compiled dependency sets (for dev, docs, pandas, polars, transformers).

## Build, Test, and Development Commands

- `uv sync --group dev`: fast dependency sync with extras.
- `ruff check` / `ruff format`: lint and auto-format (run before PRs).
- `pytest`: run the test suite.
- `python -m build`: build the package artifacts.
- `pre-commit install`: enable git hooks (optional, but recommended).

## Coding Style & Naming Conventions

- Python 3.10+ codebase, 4-space indentation, line length 88 (Ruff defaults).
- Follow Ruff linting rules (`pyproject.toml`), with tests allowing `assert` usage.
- Prefer descriptive names; avoid short, cryptic identifiers in new code.

## Testing Guidelines

- Frameworks: `pytest` and `pytest-asyncio`.
- Place tests in `tests/`; name files `test_*.py`.
- Run locally with `pytest` before opening a PR (CI runs lint + integration tests).

## Commit & Pull Request Guidelines

- Recent commit history favors `SDK-<id>: <summary>` (e.g., `SDK-78: Migrate to basedpyright`).
- Include issue/PR references when available (e.g., `(#190)`).
- PRs should describe changes clearly and confirm `ruff check` and `ruff format` passed.

## Security & Configuration Tips

- Supported Python versions: CPython 3.10–3.13.
