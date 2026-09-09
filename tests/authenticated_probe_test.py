from urllib.parse import urlsplit

import pytest
from hirundo._env import API_HOST, API_KEY
from hirundo._headers import HIRUNDO_API_VERSION, get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import READ_TIMEOUT


def test_authenticated_https_probe() -> None:
    if not API_KEY or API_KEY == "synthetic-replay-key":
        pytest.skip("A real API key is required for the authenticated HTTPS probe")
    assert urlsplit(API_HOST).scheme == "https"

    response = requests.get(
        f"{API_HOST}/organization/",
        headers=get_headers(),
        timeout=READ_TIMEOUT,
    )
    raise_for_status_with_reason(response)

    assert response.request.headers["HIRUNDO-API-VERSION"] == HIRUNDO_API_VERSION
    assert response.request.headers["Authorization"].startswith("Bearer ")
    assert isinstance(response.json(), list)
