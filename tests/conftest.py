import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.recording.support import VcrConfiguration, VcrController

# Tests use inert credentials unless the caller explicitly opts into live requests.
# This prevents a developer's ambient shell or dotenv values from reaching the network
# during ordinary unit collection or replay.
if os.environ.get("HIRUNDO_LIVE_TESTS") != "1":
    os.environ.update(
        {
            "HIRUNDO_API_HOST": "https://api.example.test",
            "HIRUNDO_API_KEY": "synthetic-replay-key",
            "API_HOST": "https://api.example.test",
            "API_KEY": "synthetic-replay-key",
            "GCP_CREDENTIALS": json.dumps({"type": "service_account"}),
            "AWS_ACCESS_KEY": "synthetic-access-key",
            "AWS_SECRET_ACCESS_KEY": "synthetic-secret-key",
            "HUGGINGFACE_ACCESS_TOKEN": "synthetic-huggingface-token",
        }
    )


RECORDED_INTEGRATION_PATHS = {
    "tests/integration/recorded_api_test.py",
}
FULL_BACKEND_PATHS = {
    "tests/classification/classification_aws_test.py",
    "tests/classification/classification_gcp_test.py",
    "tests/classification/sanity_gcp_test.py",
    "tests/get_by_name_test.py",
    "tests/llm-behavior-eval/llm_behavior_eval_test.py",
    "tests/object-detection/od_aws_test.py",
    "tests/object-detection/od_git_test.py",
    "tests/object-detection/rockpaperscisssors_yolo_test.py",
    "tests/object-detection/sama_coco_test.py",
    "tests/object-detection/sanity_aws_test.py",
    "tests/speech-to-text/sanity_stt_git_test.py",
    "tests/speech-to-text/stt_git_test.py",
    "tests/unlearning-llm/unlearn_llm_behavior_test.py",
}
AUTHENTICATED_PROBE_PATHS = {
    "tests/authenticated_probe_test.py",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "unit: tests with no network or deployed service dependency",
        "recorded_integration: real SDK integration recorded once and replayed",
        "local_transport: localhost-only transport behavior",
        "authenticated_probe: one inexpensive authenticated HTTPS request",
        "full_backend: opt-in inference and dataset pipelines that stay live",
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
        elif relative_path in FULL_BACKEND_PATHS:
            item.add_marker(pytest.mark.full_backend)
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
