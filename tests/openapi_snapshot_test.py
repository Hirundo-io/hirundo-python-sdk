import json
from pathlib import Path

import pytest
from pydantic import JsonValue
from scripts.fetch_openapi_snapshot import fetch_openapi_snapshot


class FakeResponse:
    status_code = 200
    reason = "OK"

    def json(self) -> dict[str, JsonValue]:
        return {"paths": {"/safe": {}}, "openapi": "3.1.0"}

    def raise_for_status(self) -> None:
        return None


def test_fetch_openapi_snapshot_writes_canonical_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_get(_url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        assert headers
        assert timeout > 0
        return FakeResponse()

    monkeypatch.setattr(
        "scripts.fetch_openapi_snapshot.requests.get",
        fake_get,
    )
    output_path = tmp_path / "openapi.json"

    digest = fetch_openapi_snapshot(output_path)

    assert digest.startswith("sha256:")
    assert json.loads(output_path.read_text(encoding="utf-8"))["openapi"] == "3.1.0"
