---
name: docs-examples
description: Write or edit Python SDK documentation examples in this repository's Sphinx docs.
---

Do not embed code blocks in `.rst` files. Put Python examples in standalone
`docs/*.py` files and include them with Sphinx `literalinclude` and
`:language: python`. Keep any shell examples in separate `.sh` files as well.
Make examples complete enough to show required imports and resource creation.

Documentation Python files participate in the repository's Ruff and Basedpyright
checks. Activate `.venv` and run `ruff check .`, `ruff format . --check`, and
`basedpyright` when changing examples. Validate documentation coverage with
`python scripts/validate_module_docs.py` when adding a public SDK module.

Read credentials from the environment rather than checking secrets into examples.
Verify examples that create remote resources with mocked HTTP; do not run their
training or evaluation jobs just to validate documentation.
