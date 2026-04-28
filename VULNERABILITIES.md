# Vulnerability Update Guide

Use this file when updating dependency floors in [pyproject.toml](pyproject.toml) to address vulnerability findings.

## Rules

1. Keep the dependency line and its vulnerability note together.
   Example:
   ```toml
   "requests>=2.33.0",
   #  ⬆️ Required to fix vulnerability CVE-2026-25645
   ```
   The up-arrow comment goes immediately below the package line, not above it.

2. Mention only the advisory that necessitates the current minimum version.
   If `authlib>=1.6.11` is required because `GHSA-jj8c-mmj3-mmgv` is the first fix at that floor, mention only that advisory.
   Do not list older advisories that are also fixed incidentally by the same newer version.

3. Do not keep stale version-history commentary.
   Keep only the final package floor and the advisory tied to that floor.

4. Treat major-version upgrades as compatibility risks.
   Before keeping a major bump such as `transformers` or `pytest`, add or run tests that exercise the affected public SDK behavior.

5. Prefer the smallest safe version bump that clears the finding.
   If a vulnerability is fixed in `46.0.6`, do not cite it as justification for `46.0.7` unless `46.0.7` is required by another active finding.

## Workflow

1. Update the minimum version in `pyproject.toml`.
2. Place the vulnerability comment immediately below the changed dependency line.
3. Recompile the dependency set from `pyproject.toml`.
   Example:
   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv pip compile pyproject.toml --group dev --group docs --group deploy --extra pandas --extra polars --extra transformers -o /tmp/hirundo-all-requirements.txt
   ```
4. Audit the compiled output.
   Example:
   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv run pip-audit --no-deps --disable-pip -r /tmp/hirundo-all-requirements.txt -f json
   ```
5. Run validation for compatibility-sensitive changes.
   Commands:
   ```bash
   .venv/bin/pytest
   .venv/bin/basedpyright
   ```

## Repo-specific notes

- This repository has many integration-heavy tests that require credentials such as `AWS_ACCESS_KEY`, `GCP_CREDENTIALS`, and `HUGGINGFACE_ACCESS_TOKEN`.
- If full `pytest` is unavailable because credentials are missing, report that explicitly and run any safe targeted tests that cover the changed behavior.
- For `transformers` changes, keep coverage on the Hugging Face pipeline path in [tests/unlearning-llm/llm_pipeline_transformers_test.py](tests/unlearning-llm/llm_pipeline_transformers_test.py).
