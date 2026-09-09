import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.recording.support import VcrConfiguration, VcrController

# Test collection must not require production credentials. Live CI jobs provide real
# values; unit and replay jobs keep the same import paths usable with inert values.
os.environ.setdefault("API_HOST", "https://api.invalid")
os.environ.setdefault("API_KEY", "synthetic-replay-key")
os.environ.setdefault("GCP_CREDENTIALS", json.dumps({"type": "service_account"}))
os.environ.setdefault("AWS_ACCESS_KEY", "synthetic-access-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "synthetic-secret-key")
os.environ.setdefault("HUGGINGFACE_ACCESS_TOKEN", "synthetic-huggingface-token")


RECORDED_INTEGRATION_PATHS = {
    "tests/integration/recorded_api_test.py",
}
AUTHENTICATED_PROBE_PATHS = {
    "tests/authenticated_probe_test.py",
}
LEGACY_BACKEND_PATH_PREFIXES = (
    "tests/classification/",
    "tests/object-detection/",
    "tests/speech-to-text/",
    "tests/llm-behavior-eval/",
)
LEGACY_BACKEND_PATHS = {
    "tests/get_by_name_test.py",
    "tests/unlearning-llm/unlearn_llm_behavior_test.py",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "unit: tests with no network or deployed service dependency",
        "recorded_integration: real SDK integration recorded once and replayed",
        "local_transport: localhost-only transport behavior",
        "authenticated_probe: one inexpensive authenticated HTTPS request",
        "legacy_backend: opt-in backend and full ML coverage outside the replay pilot",
    ):
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign every test to one explicit CI selection.

    Args:
        items: Collected pytest items to classify.

    Returns:
        None.
    """
    for item in items:
        relative_path = (
            Path(str(item.path)).resolve().relative_to(Path.cwd()).as_posix()
        )
        if relative_path in RECORDED_INTEGRATION_PATHS:
            item.add_marker(pytest.mark.recorded_integration)
            item.add_marker(pytest.mark.vcr)
        elif relative_path in AUTHENTICATED_PROBE_PATHS:
            item.add_marker(pytest.mark.authenticated_probe)
        elif relative_path.startswith("tests/transport/"):
            item.add_marker(pytest.mark.local_transport)
        elif relative_path in LEGACY_BACKEND_PATHS or relative_path.startswith(
            LEGACY_BACKEND_PATH_PREFIXES
        ):
            item.add_marker(pytest.mark.legacy_backend)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_recording_configure(config: pytest.Config, vcr: "VcrController") -> None:
    """Install the SDK's semantic matcher on pytest-recording's VCR instance.

    Args:
        config: Active pytest configuration.
        vcr: VCR instance created by pytest-recording.

    Returns:
        None.
    """
    from tests.recording.support import configure_vcr

    configure_vcr(vcr)


@pytest.fixture
def vcr_config() -> "VcrConfiguration":
    from tests.recording.support import vcr_config as strict_vcr_config

    return strict_vcr_config()


@pytest.fixture(autouse=True)
def independently_block_replay_network(
    request: pytest.FixtureRequest, record_mode: str
) -> Generator[None, None, None]:
    """Block sockets separately from VCR whenever an integration test replays.

    Args:
        request: Request for the test currently being executed.
        record_mode: Active pytest-recording mode.

    Returns:
        A fixture generator that restores socket access after the test.
    """
    if (
        request.node.get_closest_marker("recorded_integration")
        and record_mode == "none"
    ):
        from tests.recording.support import block_replay_network

        with block_replay_network():
            yield
    else:
        yield


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    configured_directory = os.getenv("HIRUNDO_CASSETTE_DIR")
    if configured_directory:
        module_name = Path(str(request.node.path)).stem
        cassette_directory = Path(configured_directory) / module_name
    else:
        module_path = Path(str(request.node.path))
        cassette_directory = module_path.parent / "cassettes" / module_path.stem
    cassette_directory.mkdir(parents=True, exist_ok=True)
    return str(cassette_directory)
