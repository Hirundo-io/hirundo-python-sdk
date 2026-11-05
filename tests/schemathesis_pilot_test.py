import importlib.util

import pytest
from scripts.run_schemathesis_pilot import (
    CONFIG_PATH,
    MAX_EXAMPLES,
    MAX_RUN_SECONDS,
    PLATFORM_REVISION,
    REQUEST_TIMEOUT_SECONDS,
    SAFE_PATHS,
    build_command,
)


def test_pilot_is_bounded_read_only_and_reports_unknown_revision() -> None:
    command = build_command("https://test/openapi.json", "https://test", "1")
    assert SAFE_PATHS[0] == "/llm-behavior-eval/run/info/{hir_run_id}"
    assert all(
        path.startswith(("/llm-behavior-eval/", "/unlearning-llm/"))
        for path in SAFE_PATHS
    )
    assert command.count("--include-method") == 1
    assert command[command.index("--include-method") + 1] == "GET"
    assert "token" not in " ".join(command).lower()
    assert str(MAX_EXAMPLES) == command[command.index("--max-examples") + 1]
    assert str(MAX_RUN_SECONDS) == command[command.index("--max-time") + 1]
    assert (
        str(REQUEST_TIMEOUT_SECONDS) == command[command.index("--request-timeout") + 1]
    )
    assert PLATFORM_REVISION == "unknown"
    assert "negative_data_rejection" in command[command.index("--checks") + 1]


@pytest.mark.skipif(
    importlib.util.find_spec("schemathesis") is None,
    reason="Schemathesis pilot dependency is not installed",
)
def test_pilot_cli_accepts_the_bounded_options() -> None:
    command = build_command("https://test/openapi.json", "https://test", "1")
    assert command[:3] == [command[0], "-m", "schemathesis"]
    assert command[3:5] == ["--config-file", str(CONFIG_PATH)]


@pytest.mark.skipif(
    importlib.util.find_spec("schemathesis") is None,
    reason="Schemathesis pilot dependency is not installed",
)
def test_pilot_config_resolves_secret_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from schemathesis.config import SchemathesisConfig

    monkeypatch.setenv("HIRUNDO_API_TOKEN", "test-secret")
    config = SchemathesisConfig.from_path(CONFIG_PATH)
    assert config.projects.default.headers == {"Authorization": "Bearer test-secret"}
