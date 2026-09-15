"""Recorded CRUD coverage for inexpensive public SaaS API operations.

Run-launch endpoints are deliberately excluded: Dataset QA requires accessing an
external dataset, while LLM unlearning and evaluation begin model processing.
Those workflows remain covered by opt-in backend tests rather than recordings.
"""

import os

from hirundo import (
    GitPlainAuth,
    GitRepo,
    HirundoCSV,
    HuggingFaceTransformersModel,
    LabelingType,
    LlmBehaviorEval,
    LlmModel,
    LlmUnlearningRun,
    ModalityType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageGit,
    StorageS3,
    StorageTypes,
)
from pydantic import JsonValue, TypeAdapter

REPOSITORY_URL = "https://github.com/Hirundo-io/hirundo-python-sdk"
HUGGINGFACE_REPOSITORY_URL = "https://huggingface.co/datasets/hirundo-io/RoboShaulTiny"
MODEL_SOURCE_NAME = "Qwen/Qwen3-0.6B"
GCP_CREDENTIALS_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _resource_name(resource_type: str) -> str:
    unique_id = os.environ.get("UNIQUE_ID", "local-recording")
    return f"sdk-http-recording-{resource_type}-{unique_id}"


def _new_git_storage_config(resource_type: str) -> StorageConfig:
    return StorageConfig(
        name=_resource_name(f"{resource_type}-storage"),
        type=StorageTypes.GIT,
        git=StorageGit(
            repo=GitRepo(
                name=_resource_name(f"{resource_type}-repository"),
                repository_url=REPOSITORY_URL,
            ),
            branch="main",
        ),
    )


def _delete_git_storage_config(storage_config: StorageConfig) -> None:
    repository = storage_config.git.repo if storage_config.git else None
    try:
        if storage_config.id is not None:
            storage_config.delete()
    finally:
        if repository and repository.id is not None:
            repository.delete()


def _assert_dataset_metadata_crud(dataset: QADataset) -> None:
    storage_config = dataset.storage_config
    assert isinstance(storage_config, StorageConfig)
    dataset_id: int | None = None
    try:
        dataset_id = dataset.create(replace_if_exists=True)
        assert QADataset.get_by_id(dataset_id).id == dataset_id
        assert QADataset.get_by_name(dataset.name).id == dataset_id
    finally:
        if dataset_id is not None:
            dataset.delete(storage_config=False)
        if storage_config.type is StorageTypes.GIT:
            _delete_git_storage_config(storage_config)
        elif storage_config.id is not None:
            storage_config.delete()


def test_git_repository_crud_sequence() -> None:
    repository = GitRepo(
        name=_resource_name("git-repository"),
        repository_url=REPOSITORY_URL,
    )
    try:
        repository_id = repository.create(replace_if_exists=True)
        by_id = GitRepo.get_by_id(repository_id)
        by_name = GitRepo.get_by_name(repository.name)
        listed_repositories = GitRepo.list()

        assert by_id.id == repository_id
        assert by_name.id == repository_id
        assert any(item.id == repository_id for item in listed_repositories)
    finally:
        if repository.id is not None:
            repository.delete()


def test_git_storage_config_crud_sequence() -> None:
    storage_config = _new_git_storage_config("storage")
    try:
        storage_config_id = storage_config.create(replace_if_exists=True)
        by_id = StorageConfig.get_by_id(storage_config_id)
        by_name = StorageConfig.get_by_name(storage_config.name, StorageTypes.GIT)
        listed_storage_configs = StorageConfig.list()

        assert by_id.id == storage_config_id
        assert by_name.id == storage_config_id
        assert any(item.id == storage_config_id for item in listed_storage_configs)
    finally:
        _delete_git_storage_config(storage_config)


def test_dataset_qa_metadata_crud_sequence() -> None:
    storage_config = _new_git_storage_config("dataset")
    git_storage = storage_config.git
    assert git_storage is not None
    dataset = QADataset(
        name=_resource_name("dataset"),
        labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
        modality=ModalityType.TABULAR,
        storage_config=storage_config,
        labeling_info=HirundoCSV(
            csv_url=git_storage.get_url("README.md"),
        ),
    )
    dataset_id: int | None = None
    try:
        dataset_id = dataset.create(replace_if_exists=True)
        by_id = QADataset.get_by_id(dataset_id)
        by_name = QADataset.get_by_name(dataset.name)
        listed_datasets = QADataset.list_datasets()
        listed_runs = QADataset.list_runs()

        assert by_id.id == dataset_id
        assert by_name.id == dataset_id
        assert any(item.id == dataset_id for item in listed_datasets)
        assert isinstance(listed_runs, list)
    finally:
        if dataset_id is not None:
            dataset.delete(storage_config=False)
        _delete_git_storage_config(storage_config)


def test_gcp_dataset_metadata_sequence() -> None:
    gcp_storage = StorageGCP(
        bucket_name="cifar1bucket",
        project="Hirundo-global",
        credentials_json=GCP_CREDENTIALS_ADAPTER.validate_json(
            os.environ["GCP_CREDENTIALS"]
        ),
    )
    _assert_dataset_metadata_crud(
        QADataset(
            name=_resource_name("gcp-dataset"),
            labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
            storage_config=StorageConfig(
                name=_resource_name("gcp-storage"),
                type=StorageTypes.GCP,
                gcp=gcp_storage,
            ),
            labeling_info=HirundoCSV(
                csv_url=gcp_storage.get_url("pytorch-cifar/data/cifar1.csv")
            ),
            data_root_url=gcp_storage.get_url("pytorch-cifar/data"),
        )
    )


def test_aws_dataset_metadata_sequence() -> None:
    s3_storage = StorageS3(
        bucket_url="s3://cifar10bucket",
        region_name="us-east-2",
        access_key_id=os.environ["AWS_ACCESS_KEY"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    _assert_dataset_metadata_crud(
        QADataset(
            name=_resource_name("aws-dataset"),
            labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
            storage_config=StorageConfig(
                name=_resource_name("aws-storage"),
                type=StorageTypes.S3,
                s3=s3_storage,
            ),
            labeling_info=HirundoCSV(
                csv_url=s3_storage.get_url("pytorch-cifar/data/cifar10.csv")
            ),
            data_root_url=s3_storage.get_url("pytorch-cifar/data"),
        )
    )


def test_huggingface_dataset_metadata_sequence() -> None:
    storage_config = StorageConfig(
        name=_resource_name("huggingface-storage"),
        type=StorageTypes.GIT,
        git=StorageGit(
            repo=GitRepo(
                name=_resource_name("huggingface-repository"),
                repository_url=HUGGINGFACE_REPOSITORY_URL,
                plain_auth=GitPlainAuth(
                    username="blewis-hir",
                    password=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
                ),
            ),
            branch="main",
        ),
    )
    git_storage = storage_config.git
    assert git_storage is not None
    _assert_dataset_metadata_crud(
        QADataset(
            name=_resource_name("huggingface-dataset"),
            modality=ModalityType.SPEECH,
            labeling_type=LabelingType.SPEECH_TO_TEXT,
            language="he",
            storage_config=storage_config,
            data_root_url=git_storage.get_url("wavs"),
            labeling_info=HirundoCSV(csv_url=git_storage.get_url("meta.csv")),
        )
    )


def test_llm_model_crud_sequence_and_run_listing() -> None:
    model = LlmModel(
        model_name=_resource_name("llm-model"),
        model_source=HuggingFaceTransformersModel(model_name=MODEL_SOURCE_NAME),
    )
    renamed_model_name = _resource_name("llm-model-renamed")
    try:
        model_id = model.create(replace_if_exists=True)
        by_id = LlmModel.get_by_id(model_id)
        by_name = LlmModel.get_by_name(model.model_name)
        listed_models = LlmModel.list()
        model.update(model_name=renamed_model_name, archive_existing_runs=False)
        updated_by_name = LlmModel.get_by_name(renamed_model_name)
        listed_runs = LlmUnlearningRun.list()

        assert by_id.id == model_id
        assert by_name.id == model_id
        assert any(item.id == model_id for item in listed_models)
        assert model.model_name == renamed_model_name
        assert updated_by_name.id == model_id
        assert isinstance(listed_runs, list)
    finally:
        if model.id is not None:
            model.delete()


def test_llm_behavior_eval_run_listing() -> None:
    assert isinstance(LlmBehaviorEval.list_runs(), list)
