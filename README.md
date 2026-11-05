# [hirundo](https://docs.hirundo.io/) · [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Hirundo-io/hirundo-client/blob/main/LICENSE) [![pypi version](https://img.shields.io/pypi/v/hirundo)](https://pypi.org/project/hirundo/)

[![Deploy to PyPI](https://github.com/Hirundo-io/hirundo-python-sdk/actions/workflows/deploy-to-pypi.yaml/badge.svg)](https://github.com/Hirundo-io/hirundo-python-sdk/actions/workflows/deploy-to-pypi.yaml) [![Deploy docs](https://github.com/Hirundo-io/hirundo-client/actions/workflows/update-docs.yaml/badge.svg)](https://github.com/Hirundo-io/hirundo-client/actions/workflows/update-docs.yaml) [![Ruff & Pyright](https://github.com/Hirundo-io/hirundo-client/actions/workflows/lint.yaml/badge.svg?event=merge_group)](https://github.com/Hirundo-io/hirundo-client/actions/workflows/lint.yaml?query=event%3Amerge_group) [![Sanity tests](https://github.com/Hirundo-io/hirundo-python-sdk/actions/workflows/pytest-sanity.yaml/badge.svg?event=merge_group)](https://github.com/Hirundo-io/hirundo-python-sdk/actions/workflows/pytest-sanity.yaml?query=event%3Amerge_group) [![Vulnerability scan](https://github.com/Hirundo-io/hirundo-client/actions/workflows/vulnerability-scan.yml/badge.svg?event=merge_group)](https://github.com/Hirundo-io/hirundo-client/actions/workflows/vulnerability-scan.yml?query=event%3Amerge_group)

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

This writes `HIRUNDO_API_KEY` (and optionally `HIRUNDO_API_HOST`) to `.env` in the current directory or `~/.hirundo.conf`.

## Telemetry

The SDK sends a small amount of usage and error data to Hirundo, so we can see
which features are used and fix errors faster. It is on by default and easy to
turn off.

### What is sent

- Your Hirundo user ID. It is the same ID the Hirundo web app uses, so it does
  not include your name or email.
- `SDK run launched`: the product (dataset QA, LLM unlearning, LLM behavior
  eval or external eval) and the run ID.
- `SDK error`: the product, the SDK function that failed (for example
  `LlmUnlearningRun.launch`), the error type (for example `HTTPError`) and the
  HTTP status code, if there is one.
- `SDK CLI command`: the `hirundo` command name only (for example `check-run`).
- With every event: the SDK version, Python version, operating system and its
  version, CPU architecture, whether the SDK runs in a notebook, and whether it
  talks to Hirundo's cloud API (`https://api.hirundo.io`).

### What is never sent

Names, emails, API keys, machine names, file paths, URLs, dataset or model
names, run names, error messages or tracebacks. IP-based location lookup is
turned off for every event.

### How it works

The first time a process uses the SDK, it checks once whether Hirundo's
PostHog analytics server can be reached. If it cannot, for example on an
air-gapped network, the SDK sends nothing for the rest of that process. All
telemetry runs on a background thread with a 2-second timeout and no retries,
so it never slows down or changes SDK calls. When the process exits, the SDK
waits at most 2 seconds to send the remaining events.

### Turn it off

Set either of these in your environment, `.env` or `~/.hirundo.conf`:

```bash
HIRUNDO_TELEMETRY_DISABLED=true
DO_NOT_TRACK=1
```

When telemetry is off, the SDK collects nothing and makes no telemetry network
calls.

### Settings

- `HIRUNDO_TELEMETRY_POSTHOG_KEY`: the PostHog project key. Telemetry is off
  when it is not set.
- `HIRUNDO_TELEMETRY_POSTHOG_HOST`: the PostHog server. Defaults to
  `https://us.i.posthog.com`.

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
