from typing import Any

import pytest
from hirundo import StorageConfig, StorageTypes


class _Response:
    status_code = 200

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def _build_storage_config_payload(**overrides: Any) -> dict[str, Any]:
    storage_config_payload = {
        "id": 456,
        "name": "Local",
        "type": StorageTypes.LOCAL,
        "organization_name": "org",
        "creator_name": "creator",
        "s3": None,
        "gcp": None,
        "git": None,
        "created_at": "2026-06-22T14:20:31.663Z",
        "updated_at": "2026-06-22T14:20:31.663Z",
    }
    storage_config_payload.update(overrides)
    return storage_config_payload


def _patch_storage_config_list(
    monkeypatch: pytest.MonkeyPatch,
    storage_config_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_params: list[dict[str, Any]] = []

    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        request_params.append(kwargs["params"])
        return _Response(storage_config_payloads)

    monkeypatch.setattr("hirundo.storage.requests.get", fake_get)
    return request_params


def test_get_local_returns_single_local_storage_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_params = _patch_storage_config_list(
        monkeypatch,
        [
            _build_storage_config_payload(id=123, type=StorageTypes.GCP),
            _build_storage_config_payload(id=456, type=StorageTypes.LOCAL),
        ],
    )

    storage_config = StorageConfig.get_local(organization_id=789)

    assert storage_config.id == 456
    assert storage_config.type == StorageTypes.LOCAL
    assert request_params == [{"storage_config_organization_id": 789}]


def test_get_local_selects_by_name_when_multiple_local_configs_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_storage_config_list(
        monkeypatch,
        [
            _build_storage_config_payload(id=456, name="Local"),
            _build_storage_config_payload(id=789, name="Local-backup"),
        ],
    )

    storage_config = StorageConfig.get_local(name="Local-backup")

    assert storage_config.id == 789


def test_get_local_rejects_ambiguous_local_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_storage_config_list(
        monkeypatch,
        [
            _build_storage_config_payload(id=456, name="Local"),
            _build_storage_config_payload(id=789, name="Local-backup"),
        ],
    )

    with pytest.raises(ValueError, match="Multiple local storage configs"):
        StorageConfig.get_local()


def test_get_local_rejects_missing_local_storage_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_storage_config_list(
        monkeypatch,
        [_build_storage_config_payload(type=StorageTypes.GCP)],
    )

    with pytest.raises(ValueError, match="No local storage config"):
        StorageConfig.get_local()


def test_get_default_local_is_alias_for_get_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_storage_config_list(
        monkeypatch,
        [_build_storage_config_payload(id=456)],
    )

    storage_config = StorageConfig.get_default_local()

    assert storage_config.id == 456
