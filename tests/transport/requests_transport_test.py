from pathlib import Path
from urllib.parse import urlsplit

import hirundo._http as http_transport
import pytest
import requests as requests_library
from hirundo import unzip
from hirundo._dataframe import has_pandas, has_polars
from hirundo._http import requests
from tests.transport.server import LocalTransportServer
from urllib3.util.retry import Retry


def test_server_listens_only_on_ipv4_loopback(
    local_transport_server: LocalTransportServer,
) -> None:
    parsed_url = urlsplit(local_transport_server.base_url)

    assert local_transport_server.server_address[0] == "127.0.0.1"
    assert parsed_url.hostname == "127.0.0.1"


@pytest.mark.skipif(
    not (has_pandas or has_polars),
    reason="Requires pandas or polars to materialize result DataFrames",
)
def test_download_and_extract_zip_over_streaming_http(
    local_transport_server: LocalTransportServer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _path_class: tmp_path))

    results = unzip.download_and_extract_zip(
        "localhost-run",
        f"{local_transport_server.base_url}/results.zip",
    )

    assert results.suspects is not None
    assert results.suspect_level_counts is not None
    assert results.warnings_and_errors is not None
    assert results.cached_zip_path.read_bytes()
    assert local_transport_server.request_counts["/results.zip"] == 1


def test_requests_read_timeout_is_enforced(
    local_transport_server: LocalTransportServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with requests_library.Session() as non_retrying_session:
        monkeypatch.setattr(http_transport, "_SESSION", non_retrying_session)

        with pytest.raises(requests_library.exceptions.ReadTimeout):
            requests.get(
                f"{local_transport_server.base_url}/slow",
                timeout=(1.0, 0.05),
            )


def test_requests_retries_429_with_retry_after(
    local_transport_server: LocalTransportServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Retry, "sleep", lambda _retry, _response=None: None)

    response = requests.get(
        f"{local_transport_server.base_url}/retry-success",
        timeout=1.0,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert local_transport_server.request_counts["/retry-success"] == 3


def test_requests_returns_last_429_after_retry_exhaustion(
    local_transport_server: LocalTransportServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Retry, "sleep", lambda _retry, _response=None: None)

    response = requests.get(
        f"{local_transport_server.base_url}/retry-exhausted",
        timeout=1.0,
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "0"
    assert local_transport_server.request_counts["/retry-exhausted"] == 10
