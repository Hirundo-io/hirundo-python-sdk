# Repository Guidelines

## Commands

Run from repo root unless noted.
Activate the local virtualenv before running any Python/uv commands: `source .venv/bin/activate`.

- Install deps (preferred): `uv sync --all-groups`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `basedpyright`
- Tests: `pytest`
- Build: `python -m build`
- Pre-commit hooks: `pre-commit install` (optional, but recommended)

## Project Structure & Module Organization

- `hirundo/` holds the SDK source (CLI entry point is `hirundo.cli:app`).
- `tests/` contains pytest-based test coverage.
- `docs/` and `source/` contain Sphinx documentation assets.
- `notebooks/` and `on_prem_test_notebook.ipynb` provide example workflows.

## Project Rules

- Tests are integration-heavy and often require credentials plus opt-in
  env flags (`FULL_TEST` / `RUN_*`). See `tests/dataset_qa_shared.py`.
- HTTP calls should use the retrying shim and error helper:
  `from hirundo._http import requests, raise_for_status_with_reason`.
- Use `hirundo.logger.get_logger(__name__)` for logging.
- Auth loads from `.env` or `~/.hirundo.conf` via `hirundo/_env.py`.
- Avoid 1-3 character variable names in new or refactored code. Use descriptive names
  even in small scopes.

## Coding Style & Naming Conventions

- Python 3.10+ codebase, 4-space indentation, line length 88 (Ruff defaults).
- Follow Ruff linting rules (`pyproject.toml`), with tests allowing `assert` usage.
- Prefer descriptive names; avoid short, cryptic identifiers in new code.

## Testing Guidelines

- Frameworks: `pytest` and `pytest-asyncio`.
- Place tests in `tests/`; name files `test_*.py`.
- Run locally with `pytest` before opening a PR (CI runs lint + integration tests).

## Pull Request Guidelines

- PR titles should be `SDK-<id>: <summary>` (e.g., `SDK-78: Migrate to basedpyright`).
- PRs should describe changes clearly and confirm `ruff check` and `ruff format` passed.

## Security & Configuration Tips

- Supported Python versions: CPython 3.10–3.13.
