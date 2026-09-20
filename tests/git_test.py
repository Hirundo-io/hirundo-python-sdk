from typing import TypeAlias

import pytest
from hirundo.git import GitRepo
from pydantic import JsonValue

JsonObject: TypeAlias = dict[str, JsonValue]


class _Response:
    status_code = 200

    def __init__(self, payload: JsonObject | list[JsonObject]) -> None:
        self.payload = payload

    def json(self) -> JsonObject | list[JsonObject]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def test_git_repo_reads_send_organization_query_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_params: list[JsonObject] = []
    repository: JsonObject = {
        "id": 7,
        "name": "repository",
        "repository_url": "https://example.test/repository.git",
        "created_at": "2026-09-15T10:00:00Z",
        "updated_at": "2026-09-15T10:00:00Z",
    }

    def fake_get(
        url: str,
        *,
        params: JsonObject,
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        request_params.append(params)
        return _Response([repository] if url.endswith("/git-repo/") else repository)

    monkeypatch.setattr("hirundo.git.requests.get", fake_get)

    GitRepo.get_by_id(7, organization_id=11)
    GitRepo.get_by_name("repository", organization_id=11)
    GitRepo.list(organization_id=11)

    assert request_params == [
        {"git_repo_organization_id": 11},
        {"git_repo_organization_id": 11},
        {"git_repo_organization_id": 11},
    ]
