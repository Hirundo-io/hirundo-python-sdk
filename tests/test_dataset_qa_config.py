from typing import Any

import pytest
from hirundo import HirundoCSV, LabelingType, ModalityType, QADataset
from pydantic_core import Url


class _Response:
    status_code = 200

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def _create_dataset_response() -> _Response:
    return _Response({"id": 123})


def _build_dataset_payload(**overrides: Any) -> dict[str, Any]:
    dataset_payload = {
        "name": "tabular dataset",
        "labeling_type": LabelingType.SINGLE_LABEL_CLASSIFICATION,
        "storage_config_id": 456,
        "data_root_url": "gs://bucket/data",
        "labeling_info": {
            "type": "HirundoCSV",
            "csv_url": "gs://bucket/data/metadata.csv",
        },
        "classes": ["ok", "bad"],
        "modality": ModalityType.TABULAR,
    }
    dataset_payload.update(overrides)
    return dataset_payload


def _build_storage_config_payload(**overrides: Any) -> dict[str, Any]:
    storage_config_payload = {
        "id": 456,
        "name": "dataset-storage",
        "type": "GCP",
        "organization_name": "org",
        "creator_name": "creator",
        "s3": None,
        "gcp": {"bucket_name": "bucket", "project": "project"},
        "git": None,
        "created_at": "2026-06-22T14:20:31.663Z",
        "updated_at": "2026-06-22T14:20:31.663Z",
    }
    storage_config_payload.update(overrides)
    return storage_config_payload


def _build_dataset(**overrides: Any) -> QADataset:
    dataset_payload = _build_dataset_payload(
        data_root_url=Url("gs://bucket/data"),
        labeling_info=HirundoCSV(csv_url=Url("gs://bucket/data/metadata.csv")),
    )
    dataset_payload.update(overrides)
    return QADataset(**dataset_payload)


def _capture_create_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    request_payloads: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs["json"])
        return _create_dataset_response()

    monkeypatch.setattr("hirundo.dataset_qa.requests.post", fake_post)
    return request_payloads


def test_tabular_extra_non_feature_cols_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_dataset(extra_non_feature_cols=["sample_id", "source"])
    dataset.create()

    assert request_payloads[0]["extra_non_feature_cols"] == ["sample_id", "source"]


def test_tabular_feature_cols_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_dataset(feature_cols=["age", "score"])
    dataset.create()

    assert request_payloads[0]["feature_cols"] == ["age", "score"]


@pytest.mark.parametrize(
    "column_option",
    (
        {"extra_non_feature_cols": ["sequence_id"]},
        {"feature_cols": ["temperature", "pressure"]},
    ),
)
def test_timeseries_column_options_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
    column_option: dict[str, list[str]],
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_dataset(
        name="timeseries dataset",
        modality=ModalityType.TIMESERIES,
        **column_option,
    )
    dataset.create()

    option_name = next(iter(column_option))
    assert request_payloads[0]["modality"] == ModalityType.TIMESERIES
    assert request_payloads[0][option_name] == column_option[option_name]


def test_timeseries_dataset_run_launches_after_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads: list[dict[str, Any] | None] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs.get("json"))
        if str(args[0]).endswith("/dataset-qa/run/123"):
            return _Response({"run_id": "run-123"})
        return _create_dataset_response()

    monkeypatch.setattr("hirundo.dataset_qa.requests.post", fake_post)

    dataset = _build_dataset(modality=ModalityType.TIMESERIES)

    assert dataset.run_qa() == "run-123"
    assert dataset.run_id == "run-123"
    create_payload = request_payloads[0]
    assert create_payload is not None
    assert create_payload["modality"] == ModalityType.TIMESERIES


@pytest.mark.parametrize("modality", (ModalityType.TABULAR, ModalityType.TIMESERIES))
def test_tabular_like_datasets_can_omit_data_root_url(
    monkeypatch: pytest.MonkeyPatch,
    modality: ModalityType,
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_dataset(data_root_url=None, modality=modality)
    dataset.create()

    assert dataset.data_root_url is None
    assert "data_root_url" not in request_payloads[0]


def test_data_root_url_is_required_for_non_tabular_like_datasets() -> None:
    with pytest.raises(ValueError, match="data_root_url"):
        _build_dataset(data_root_url=None, modality=ModalityType.VISION)


def test_tabular_column_options_default_to_none() -> None:
    dataset = _build_dataset()

    assert dataset.extra_non_feature_cols is None
    assert dataset.feature_cols is None


def test_tabular_empty_column_options_are_normalized_to_none() -> None:
    dataset = _build_dataset(extra_non_feature_cols=[], feature_cols=[])

    assert dataset.extra_non_feature_cols is None
    assert dataset.feature_cols is None


def test_tabular_column_options_are_omitted_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_dataset()
    dataset.create()

    assert "extra_non_feature_cols" not in request_payloads[0]
    assert "feature_cols" not in request_payloads[0]


@pytest.mark.parametrize(
    "column_option",
    (
        {"extra_non_feature_cols": ["sample_id"]},
        {"feature_cols": ["age"]},
    ),
)
def test_tabular_column_options_are_rejected_for_non_tabular_datasets(
    column_option: dict[str, list[str]],
) -> None:
    with pytest.raises(ValueError, match="tabular or timeseries datasets"):
        _build_dataset(
            **column_option,
            modality=ModalityType.VISION,
        )


def test_tabular_column_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Only one"):
        _build_dataset(
            extra_non_feature_cols=["sample_id"],
            feature_cols=["age"],
        )


def test_get_by_id_accepts_empty_tabular_column_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            _build_dataset_payload(
                id=123,
                extra_non_feature_cols=[],
                feature_cols=[],
            )
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.get_by_id(123)

    assert str(dataset.data_root_url) == "gs://bucket/data"
    assert dataset.extra_non_feature_cols is None
    assert dataset.feature_cols is None


def test_get_by_name_accepts_matching_storage_config_and_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            _build_dataset_payload(
                id=123,
                storage_config_id=456,
                storage_config=_build_storage_config_payload(id=456),
            )
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.get_by_name("tabular dataset")

    assert dataset.storage_config_id == 456


def test_get_by_id_accepts_timeseries_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            _build_dataset_payload(
                id=123,
                modality="TIMESERIES",
                data_root_url=None,
                extra_non_feature_cols=["sequence_id"],
            )
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.get_by_id(123)

    assert dataset.modality == ModalityType.TIMESERIES
    assert dataset.data_root_url is None
    assert dataset.extra_non_feature_cols == ["sequence_id"]


def test_list_datasets_accepts_timeseries_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            [
                _build_dataset_payload(
                    id=123,
                    modality=ModalityType.TIMESERIES,
                    data_root_url=None,
                    storage_config=_build_storage_config_payload(
                        name="timeseries-storage"
                    ),
                    organization_id=789,
                    creator_id=321,
                    created_at="2026-06-22T14:20:31.663Z",
                    updated_at="2026-06-22T14:20:31.663Z",
                )
            ]
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.list_datasets()[0]

    assert dataset.modality == ModalityType.TIMESERIES
    assert dataset.storage_config_id == 456
    assert dataset.data_root_url is None


def test_list_runs_accepts_timeseries_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            [
                {
                    "id": 1,
                    "name": "timeseries run",
                    "dataset_id": 123,
                    "run_id": "run-123",
                    "modality": "TIMESERIES",
                    "status": "PENDING",
                    "approved": False,
                    "created_at": "2026-06-22T14:20:31.663Z",
                    "run_args": None,
                }
            ]
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    run = QADataset.list_runs()[0]

    assert run.modality == ModalityType.TIMESERIES
