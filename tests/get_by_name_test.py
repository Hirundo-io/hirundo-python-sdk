import json
import os
import typing

import pytest
from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageGit,
    StorageTypes,
)
from tests.testing_utils import get_unique_id

unique_id = get_unique_id()
gcp_storage_config_name = f"T-cifar1bucket_get_by_name{unique_id}"
gcp_qa_dataset_name = f"T-cifar1_get_by_name{unique_id}"
git_storage_config_name = f"T-BDD-100k-validation-git_get_by_name{unique_id}"
git_repository_name = f"T-BDD-100k-validation-git-repo_get_by_name{unique_id}"
git_qa_dataset_name = f"T-BDD-100k-validation-dataset_get_by_name{unique_id}"

new_storage_config: typing.Optional[StorageConfig] = None
new_dataset = None


@pytest.fixture(autouse=True)
def cleanup_tests():
    yield
    if new_dataset:
        new_dataset.delete()
    if new_storage_config and new_storage_config.git and new_storage_config.git.repo:
        new_storage_config.git.repo.delete()
    if new_storage_config:
        new_storage_config.delete()


def test_get_by_name_gcp():
    StorageConfig(
        name=gcp_storage_config_name,
        type=StorageTypes.GCP,
        gcp=StorageGCP(
            bucket_name="cifar1bucket",
            project="Hirundo-global",
            credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
        ),
    ).create(replace_if_exists=True)

    new_storage_config = StorageConfig.get_by_name(
        gcp_storage_config_name, StorageTypes.GCP
    )

    assert new_storage_config is not None
    assert new_storage_config.gcp is not None
    storage_gcp = new_storage_config.gcp

    QADataset(
        name=gcp_qa_dataset_name,
        labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
        storage_config=new_storage_config,
        labeling_info=HirundoCSV(
            csv_url=storage_gcp.get_url("/pytorch-cifar/data/cifar1.csv"),
        ),
        data_root_url=storage_gcp.get_url("/pytorch-cifar/data"),
    ).create(replace_if_exists=True)

    dataset = QADataset.get_by_name(gcp_qa_dataset_name)
    assert dataset is not None


def test_get_by_name_git():
    StorageConfig(
        name=git_storage_config_name,
        type=StorageTypes.GIT,
        git=StorageGit(
            repo=GitRepo(
                name=git_repository_name,
                repository_url="https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only",
            ),
            branch="main",
        ),
    ).create(replace_if_exists=True)
    new_storage_config = StorageConfig.get_by_name(
        git_storage_config_name, StorageTypes.GIT
    )

    assert new_storage_config is not None
    assert new_storage_config.git is not None
    storage_git = new_storage_config.git

    QADataset(
        name=git_qa_dataset_name,
        labeling_type=LabelingType.OBJECT_DETECTION,
        storage_config=new_storage_config,
        labeling_info=HirundoCSV(
            csv_url=storage_git.get_url(
                path="/BDD100K Val from Hirundo.zip/bdd100k/bdd100k.csv"
            )
        ),
        data_root_url=storage_git.get_url(path="/BDD100K Val from Hirundo.zip/bdd100k"),
    ).create(replace_if_exists=True)

    new_dataset = QADataset.get_by_name(git_qa_dataset_name)
    assert new_dataset is not None
