from typing import Any

import pytest
from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    ModalityType,
    MultimodalHirundoCSV,
    MultimodalModalityCSV,
    MultimodalModalityType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageGit,
    StorageTypes,
)
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


def _capture_create_and_run_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any] | None]:
    request_payloads: list[dict[str, Any] | None] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs.get("json"))
        if str(args[0]).endswith("/dataset-qa/run/123"):
            return _Response({"run_id": "run-123"})
        return _create_dataset_response()

    monkeypatch.setattr("hirundo.dataset_qa.requests.post", fake_post)
    return request_payloads


def _capture_storage_and_dataset_create_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    request_payloads: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs["json"])
        if str(args[0]).endswith("/storage-config/"):
            return _Response({"id": 456})
        return _create_dataset_response()

    monkeypatch.setattr("hirundo.storage.requests.post", fake_post)
    monkeypatch.setattr("hirundo.dataset_qa.requests.post", fake_post)
    return request_payloads


def _capture_git_and_storage_create_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    request_payloads: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        request_payloads.append(kwargs["json"])
        if str(args[0]).endswith("/git-repo/"):
            return _Response({"id": 321})
        return _Response({"id": 456})

    monkeypatch.setattr("hirundo.git.requests.post", fake_post)
    monkeypatch.setattr("hirundo.storage.requests.post", fake_post)
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


def test_dataset_create_propagates_organization_to_storage_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_storage_and_dataset_create_payloads(monkeypatch)
    dataset = _build_dataset(
        storage_config_id=None,
        storage_config=StorageConfig(
            name="dataset-storage",
            type=StorageTypes.GCP,
            gcp=StorageGCP(bucket_name="bucket", project="project"),
        ),
    )

    dataset.create(organization_id=4)

    storage_payload = request_payloads[0]
    dataset_payload = request_payloads[1]
    assert storage_payload["organization_id"] == 4
    assert dataset_payload["organization_id"] == 4
    assert dataset_payload["storage_config_id"] == 456


def test_storage_config_create_propagates_organization_to_git_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_git_and_storage_create_payloads(monkeypatch)
    storage_config = StorageConfig(
        name="dataset-git-storage",
        type=StorageTypes.GIT,
        git=StorageGit(
            branch="main",
            repo=GitRepo(
                name="dataset-repo",
                repository_url="https://example.com/org/repo.git",
            ),
        ),
    )

    assert storage_config.create(organization_id=4) == 456

    git_payload = request_payloads[0]
    storage_payload = request_payloads[1]
    assert git_payload["organization_id"] == 4
    assert storage_payload["organization_id"] == 4
    assert storage_payload["git"]["repo_id"] == 321


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
    request_payloads = _capture_create_and_run_payloads(monkeypatch)
    dataset = _build_dataset(modality=ModalityType.TIMESERIES)

    assert dataset.run_qa() == "run-123"
    assert dataset.run_id == "run-123"
    create_payload = request_payloads[0]
    assert create_payload is not None
    assert create_payload["modality"] == ModalityType.TIMESERIES


def _build_multimodal_labeling_info() -> MultimodalHirundoCSV:
    return MultimodalHirundoCSV(
        modality_csvs=[
            MultimodalModalityCSV(
                modality=MultimodalModalityType.VISION,
                labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image.csv")),
                data_root_url=Url("gs://bucket/images"),
                augmentations=["RandomHorizontalFlip"],
            ),
            MultimodalModalityCSV(
                modality=MultimodalModalityType.TABULAR,
                labeling_info=HirundoCSV(csv_url=Url("gs://bucket/tabular.csv")),
                extra_non_feature_cols=["sample_id"],
            ),
        ],
        alignment_csv_url=Url("gs://bucket/alignment.csv"),
    )


def _build_multimodal_labeling_info_payload() -> dict[str, Any]:
    return _build_multimodal_labeling_info().model_dump(mode="json")


def _build_multimodal_dataset(**overrides: Any) -> QADataset:
    dataset_payload = {
        "name": "multimodal dataset",
        "modality": ModalityType.MULTIMODAL,
        "data_root_url": None,
        "labeling_info": _build_multimodal_labeling_info(),
    }
    dataset_payload.update(overrides)
    return _build_dataset(**dataset_payload)


def test_multimodal_dataset_creation_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_payload(monkeypatch)

    dataset = _build_multimodal_dataset()
    dataset.create()

    create_payload = request_payloads[0]
    assert create_payload["modality"] == ModalityType.MULTIMODAL
    assert "data_root_url" not in create_payload
    assert "augmentations" not in create_payload
    assert create_payload["labeling_info"] == _build_multimodal_labeling_info_payload()


def test_multimodal_dataset_run_launches_after_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_and_run_payloads(monkeypatch)
    dataset = _build_multimodal_dataset()

    assert dataset.run_qa(organization_id=4) == "run-123"
    assert dataset.run_id == "run-123"
    create_payload = request_payloads[0]
    run_payload = request_payloads[1]
    assert create_payload is not None
    assert create_payload["organization_id"] == 4
    assert create_payload["modality"] == ModalityType.MULTIMODAL
    assert run_payload == {"organization_id": 4, "run_args": {}}


def test_launch_qa_run_includes_default_run_args_with_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads = _capture_create_and_run_payloads(monkeypatch)

    assert QADataset.launch_qa_run(123, organization_id=4) == "run-123"

    assert request_payloads == [{"organization_id": 4, "run_args": {}}]


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


def test_multimodal_file_child_requires_data_root_url() -> None:
    labeling_info = MultimodalHirundoCSV(
        modality_csvs=[
            MultimodalModalityCSV(
                modality=MultimodalModalityType.VISION,
                labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image.csv")),
            ),
            MultimodalModalityCSV(
                modality=MultimodalModalityType.TABULAR,
                labeling_info=HirundoCSV(csv_url=Url("gs://bucket/tabular.csv")),
            ),
        ]
    )

    with pytest.raises(ValueError, match="VISION"):
        _build_dataset(
            modality=ModalityType.MULTIMODAL,
            data_root_url=None,
            labeling_info=labeling_info,
        )


def test_multimodal_rejects_top_level_data_root_url() -> None:
    with pytest.raises(ValueError, match="top-level `data_root_url`"):
        _build_multimodal_dataset(data_root_url=Url("gs://bucket/images"))


@pytest.mark.parametrize(
    "modality",
    (MultimodalModalityType.TABULAR, MultimodalModalityType.TIMESERIES),
)
def test_multimodal_tabular_like_child_rejects_data_root_url(
    modality: MultimodalModalityType,
) -> None:
    with pytest.raises(ValueError, match="vision or radar child modalities"):
        MultimodalModalityCSV(
            modality=modality,
            labeling_info=HirundoCSV(csv_url=Url("gs://bucket/metadata.csv")),
            data_root_url=Url("gs://bucket/data"),
        )


def test_multimodal_rejects_dataset_level_augmentations() -> None:
    with pytest.raises(ValueError, match="dataset-level augmentations"):
        _build_multimodal_dataset(
            augmentations=["RandomHorizontalFlip"],
        )


def test_multimodal_rejects_top_level_column_options() -> None:
    with pytest.raises(ValueError, match="tabular or timeseries datasets"):
        _build_multimodal_dataset(feature_cols=["age"])


def test_multimodal_child_empty_column_options_are_normalized() -> None:
    modality_csv = MultimodalModalityCSV(
        modality=MultimodalModalityType.TABULAR,
        labeling_info=HirundoCSV(csv_url=Url("gs://bucket/tabular.csv")),
        feature_cols=[],
        extra_non_feature_cols=[],
    )

    assert modality_csv.feature_cols is None
    assert modality_csv.extra_non_feature_cols is None


def test_multimodal_child_rejects_unknown_augmentation() -> None:
    with pytest.raises(ValueError, match="TypoAugmentation"):
        MultimodalModalityCSV(
            modality=MultimodalModalityType.VISION,
            labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image.csv")),
            data_root_url=Url("gs://bucket/images"),
            augmentations=["TypoAugmentation"],
        )


def test_multimodal_child_rejects_augmentation_for_wrong_modality() -> None:
    with pytest.raises(ValueError, match="TABULAR"):
        MultimodalModalityCSV(
            modality=MultimodalModalityType.TABULAR,
            labeling_info=HirundoCSV(csv_url=Url("gs://bucket/tabular.csv")),
            augmentations=["RandomHorizontalFlip"],
        )


def test_multimodal_child_accepts_legacy_image_modality() -> None:
    modality_csv = MultimodalModalityCSV.model_validate(
        {
            "modality": "IMAGE",
            "labeling_info": {
                "type": "HirundoCSV",
                "csv_url": "gs://bucket/image.csv",
            },
        }
    )

    assert modality_csv.modality == MultimodalModalityType.VISION


def test_multimodal_labeling_info_requires_at_least_two_modalities() -> None:
    with pytest.raises(ValueError, match="at least two modalities"):
        MultimodalHirundoCSV(
            modality_csvs=[
                MultimodalModalityCSV(
                    modality=MultimodalModalityType.VISION,
                    labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image.csv")),
                    data_root_url=Url("gs://bucket/images"),
                )
            ]
        )


def test_multimodal_labeling_info_rejects_duplicate_modalities() -> None:
    with pytest.raises(ValueError, match="modalities must be unique"):
        MultimodalHirundoCSV(
            modality_csvs=[
                MultimodalModalityCSV(
                    modality=MultimodalModalityType.VISION,
                    labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image.csv")),
                    data_root_url=Url("gs://bucket/images"),
                ),
                MultimodalModalityCSV(
                    modality=MultimodalModalityType.VISION,
                    labeling_info=HirundoCSV(csv_url=Url("gs://bucket/image-2.csv")),
                    data_root_url=Url("gs://bucket/images-2"),
                ),
            ]
        )


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


def test_get_by_id_accepts_multimodal_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            _build_dataset_payload(
                id=123,
                name="multimodal dataset",
                modality="MULTIMODAL",
                data_root_url=None,
                labeling_info=_build_multimodal_labeling_info_payload(),
            )
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.get_by_id(123)

    assert dataset.modality == ModalityType.MULTIMODAL
    assert isinstance(dataset.labeling_info, MultimodalHirundoCSV)
    assert dataset.labeling_info.modality_csvs[0].modality == (
        MultimodalModalityType.VISION
    )
    assert dataset.labeling_info.modality_csvs[1].extra_non_feature_cols == [
        "sample_id"
    ]


def test_get_by_id_accepts_multimodal_child_empty_column_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labeling_info = _build_multimodal_labeling_info_payload()
    labeling_info["modality_csvs"][1]["feature_cols"] = []
    labeling_info["modality_csvs"][1]["extra_non_feature_cols"] = []

    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            _build_dataset_payload(
                id=123,
                name="multimodal dataset",
                modality="MULTIMODAL",
                data_root_url=None,
                labeling_info=labeling_info,
            )
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    dataset = QADataset.get_by_id(123)

    assert isinstance(dataset.labeling_info, MultimodalHirundoCSV)
    tabular_child = dataset.labeling_info.modality_csvs[1]
    assert tabular_child.feature_cols is None
    assert tabular_child.extra_non_feature_cols is None


def test_multimodal_labeling_info_list_is_rejected_for_non_multimodal_dataset() -> None:
    with pytest.raises(ValueError, match="MULTIMODAL datasets"):
        _build_dataset(
            modality=ModalityType.VISION,
            labeling_info=[_build_multimodal_labeling_info()],
        )


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


def test_list_datasets_accepts_multimodal_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            [
                _build_dataset_payload(
                    id=123,
                    name="multimodal dataset",
                    modality=ModalityType.MULTIMODAL,
                    data_root_url=None,
                    labeling_info=_build_multimodal_labeling_info_payload(),
                    storage_config=_build_storage_config_payload(
                        name="multimodal-storage"
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

    assert dataset.modality == ModalityType.MULTIMODAL
    assert isinstance(dataset.labeling_info, MultimodalHirundoCSV)
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


def test_list_runs_accepts_multimodal_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _Response:
        return _Response(
            [
                {
                    "id": 1,
                    "name": "multimodal run",
                    "dataset_id": 123,
                    "run_id": "run-123",
                    "modality": "MULTIMODAL",
                    "status": "PENDING",
                    "approved": False,
                    "created_at": "2026-06-22T14:20:31.663Z",
                    "run_args": None,
                }
            ]
        )

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    run = QADataset.list_runs()[0]

    assert run.modality == ModalityType.MULTIMODAL
