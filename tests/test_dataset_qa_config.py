from typing import Any

import pytest
from hirundo import HirundoCSV, LabelingType, ModalityType, QADataset
from pydantic_core import Url


class _CreateDatasetResponse:
    status_code = 200

    def json(self) -> dict[str, int]:
        return {"id": 123}

    def raise_for_status(self) -> None:
        return None


def _build_dataset(**overrides: Any) -> QADataset:
    dataset_kwargs = {
        "name": "tabular dataset",
        "labeling_type": LabelingType.SINGLE_LABEL_CLASSIFICATION,
        "storage_config_id": 456,
        "data_root_url": Url("gs://bucket/data"),
        "labeling_info": HirundoCSV(csv_url=Url("gs://bucket/data/metadata.csv")),
        "classes": ["ok", "bad"],
        "modality": ModalityType.TABULAR,
    }
    dataset_kwargs.update(overrides)
    return QADataset(**dataset_kwargs)


def _capture_create_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    request_payloads: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _CreateDatasetResponse:
        request_payloads.append(kwargs["json"])
        return _CreateDatasetResponse()

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


def test_tabular_column_options_default_to_none() -> None:
    dataset = _build_dataset()

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
    with pytest.raises(ValueError, match="tabular datasets"):
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
