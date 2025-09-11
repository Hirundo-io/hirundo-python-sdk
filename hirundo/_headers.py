from hirundo._env import API_KEY, check_api_key

HIRUNDO_API_VERSION = "0.3"

_json_headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _get_auth_headers():
    check_api_key()
    return {
        "Authorization": f"Bearer {API_KEY}",
    }


def _get_api_version_header():
    return {
        "HIRUNDO-API-VERSION": HIRUNDO_API_VERSION,
    }


def get_headers():
    return {
        **_json_headers,
        **_get_auth_headers(),
        **_get_api_version_header(),
    }
