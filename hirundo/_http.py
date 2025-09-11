import requests as _requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import hirundo.logger

logger = hirundo.logger.get_logger(__name__)

MINIMUM_CLIENT_SERVER_ERROR_CODE = 400


def _build_retrying_session() -> _requests.Session:
    # No more than 10 tries total (including the initial attempt)
    # urllib3 Retry.total counts retries, not total attempts, so use 9 retries
    retries = Retry(
        total=9,
        backoff_factor=1.0,
        status_forcelist=(429,),
        allowed_methods=("HEAD", "GET", "PUT", "POST", "PATCH", "DELETE", "OPTIONS"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = _requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _build_retrying_session()


class _RequestsShim:
    """Shim exposing a subset of the requests API but backed by a retrying Session."""

    HTTPError = _requests.HTTPError
    Response = _requests.Response

    def request(self, method: str, url: str, **kwargs) -> Response:
        return _SESSION.request(method=method, url=url, **kwargs)

    def get(self, url: str, **kwargs) -> Response:
        return _SESSION.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> Response:
        return _SESSION.post(url, **kwargs)

    def delete(self, url: str, **kwargs) -> Response:
        return _SESSION.delete(url, **kwargs)

    def patch(self, url: str, **kwargs) -> Response:
        return _SESSION.patch(url, **kwargs)

    def put(self, url: str, **kwargs) -> Response:
        return _SESSION.put(url, **kwargs)


# Public shim to be imported by modules instead of the raw requests package
requests = _RequestsShim()


def raise_for_status_with_reason(response: Response):
    try:
        if response.status_code >= MINIMUM_CLIENT_SERVER_ERROR_CODE:
            response.reason = response.json().get("reason", None)
            if response.reason is None:
                response.reason = response.json().get("detail", None)
    except Exception as e:
        logger.debug("Could not parse response as JSON: %s", e)

    response.raise_for_status()
