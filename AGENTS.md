# Commands

Run from repo root unless noted.
Activate the local virtualenv before running any Python/uv commands: `source .venv/bin/activate`.

- Install deps (preferred): `uv sync --all-groups`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `basedpyright`
- Tests: `pytest`

# Project rules

- Tests are integration-heavy and often require credentials plus opt-in
  env flags (`FULL_TEST` / `RUN_*`). See `tests/dataset_qa_shared.py`.
- HTTP calls should use the retrying shim and error helper:
  `from hirundo._http import requests, raise_for_status_with_reason`.
- Use `hirundo.logger.get_logger(__name__)` for logging.
- Auth loads from `.env` or `~/.hirundo.conf` via `hirundo/_env.py`.
- Avoid 1-3 character variable names in new or refactored code. Use descriptive names even in small scopes.
