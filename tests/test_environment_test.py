import json
import os
import subprocess
import sys

import pytest
from tests.recording.support import INERT_API_ORIGIN


@pytest.mark.parametrize("live_tests", ["", "0", "1"])
def test_test_environment_uses_cassette_origin_unless_live_opted_in(
    live_tests: str,
) -> None:
    environment = {
        **os.environ,
        "HIRUNDO_LIVE_TESTS": live_tests,
        "HIRUNDO_API_HOST": "https://ambient.example.test",
        "API_HOST": "https://legacy.example.test",
        "HIRUNDO_API_KEY": "ambient-test-key",
        "API_KEY": "legacy-test-key",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os, tests.conftest; from hirundo import _env; "
            "print(json.dumps([_env.API_HOST, _env.API_KEY, "
            "os.environ['API_HOST'], os.environ['API_KEY']]))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = (
        [
            "https://ambient.example.test",
            "ambient-test-key",
            "https://legacy.example.test",
            "legacy-test-key",
        ]
        if live_tests == "1"
        else [
            INERT_API_ORIGIN,
            "synthetic-replay-key",
            INERT_API_ORIGIN,
            "synthetic-replay-key",
        ]
    )
    assert json.loads(result.stdout) == expected
