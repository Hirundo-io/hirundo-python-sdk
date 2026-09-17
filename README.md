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

The CLI stores the API key in the operating system's credential store and keeps
the API host in `.env` or `~/.hirundo.conf`. On Linux, a keyring usually requires
a desktop Secret Service or KWallet session. When no usable keyring is available,
`auto` mode warns and stores the key in the selected configuration file with
owner-only permissions.

For CI, containers, SSH sessions, and headless servers, avoid persistent local
credentials and provide `HIRUNDO_API_KEY` through the environment or your secret
manager. Environment variables take precedence over the keyring and configuration
files.

Use `--key-storage keyring` to require secure storage and fail if no backend is
available. Hirundo accepts the native macOS Keychain, Windows Credential Locker,
Linux Secret Service, KWallet, and libsecret backends; plaintext and unknown
third-party keyring backends are rejected. Use `--key-storage file` to choose the
private-file fallback explicitly:

```bash
hirundo setup --key-storage keyring
hirundo set-api-key --key-storage file
```

When a key is saved successfully to the keyring, the CLI removes
`HIRUNDO_API_KEY` from the active configuration file. Existing configuration
files remain readable for backward compatibility. Conversely, selecting file
storage removes any older keyring entry for the same normalized API host so a
stale key cannot take precedence.

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
