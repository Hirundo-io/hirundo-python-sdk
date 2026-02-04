# Hirundo Python SDK

The Hirundo Python SDK lets you:

- Launch and monitor LLM behavior unlearning runs.
- Run LLM behavior evaluations for bias, hallucination, and prompt injection.
- Run dataset QA for ML datasets (classification, object detection, and more).
- Fetch QA results as `pandas` or `polars` DataFrames.

This SDK requires access to a Hirundo server (SaaS, VPC, or on-prem).

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13 (CPython).
- A Hirundo API key.

## Installation

```bash
pip install hirundo
```

Optional extras:

- LLM behavior unlearning (Transformers + PEFT): `pip install hirundo[transformers]`
- Dataset QA or LLM behavior eval results as DataFrames: `pip install hirundo[pandas]` or `pip install hirundo[polars]`

If you want to install from source, clone this repository and run:

```bash
pip install .
```

## Configure API access

You can set environment variables directly or use the CLI helper:

```bash
hirundo setup
```

This writes `API_KEY` (and optionally `API_HOST`) to `.env` in the current directory or `~/.hirundo.conf`.

## Quickstart examples

The full quickstart examples now live in the Sphinx docs so they can be linted,
formatted, and type-checked as real Python files. See the examples embedded in
`docs/index.rst`, which are sourced from `docs/*.py` files.

## Supported dataset storage

- Amazon S3
- Google Cloud Storage (GCS)
- Git repositories with LFS (GitHub, Hugging Face)

## Further documentation

- Documentation site: [https://docs.hirundo.io/](https://docs.hirundo.io/)
- Example notebooks: [notebooks/](notebooks/)
