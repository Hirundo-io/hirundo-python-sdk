import datetime
from datetime import timedelta, timezone
from typing import Union

import requests
from hirundo import GitRepo, QADataset, StorageConfig
from hirundo.dataset_qa import DataQARunOut, HirundoError, QADatasetOut, RunStatus
from hirundo.logger import get_logger
from hirundo.storage import ResponseStorageConfig

logger = get_logger(__name__)


def _delete_dataset(
    dataset_id: int,
    storage_config: Union[StorageConfig, ResponseStorageConfig, None],
    deleted_datasets: set[int],
    deleted_storage_configs: set[int],
    deleted_git_repos: set[int],
) -> tuple[set[int], set[int], set[int]]:
    try:
        QADataset.delete_by_id(dataset_id)
        deleted_datasets.add(dataset_id)
    except (HirundoError, requests.HTTPError) as exc:
        logger.warning("Failed to delete dataset with ID %s: %s", dataset_id, exc)

    if storage_config and storage_config.id is not None:
        try:
            StorageConfig.delete_by_id(storage_config.id)
            deleted_storage_configs.add(storage_config.id)
        except (HirundoError, requests.HTTPError) as exc:
            logger.warning(
                "Failed to delete storage config with ID %s: %s", storage_config.id, exc
            )

        if (
            storage_config.git is not None
            and storage_config.git.repo is not None
            and storage_config.git.repo.id is not None
        ):
            git_repo_id = storage_config.git.repo.id
            try:
                GitRepo.delete_by_id(git_repo_id)
                deleted_git_repos.add(git_repo_id)
            except (HirundoError, requests.HTTPError) as exc:
                logger.warning(
                    "Failed to delete git repo with ID %s: %s", git_repo_id, exc
                )
    return deleted_datasets, deleted_storage_configs, deleted_git_repos


def _should_delete_dataset(
    dataset_name: str, dataset_runs: list, expiry_date: datetime.datetime
) -> bool:
    """
    Return ``True`` if the dataset should be deleted.

    Args:
        dataset_name: The name of the dataset.
        dataset_runs: The runs of the dataset.
        expiry_date: The expiry date of the dataset.
    """

    if not (dataset_name.startswith("TEST-") or dataset_name.startswith("T-")):
        return False

    all_runs_successful_or_archived = all(
        run.status == RunStatus.SUCCESS or run.deleted_at is not None
        for run in dataset_runs
    )
    if all_runs_successful_or_archived:
        return True

    most_recent_run_time = max(run.created_at for run in dataset_runs)
    return most_recent_run_time <= expiry_date


def _collect_runs_by_dataset(
    datasets: dict[int, QADatasetOut],
    all_live_runs: list[DataQARunOut],
    all_archived_runs: list[DataQARunOut],
) -> dict[int, list]:
    runs_by_dataset: dict[int, list] = dict()
    for dataset_id in datasets.keys():
        runs_by_dataset[dataset_id] = []
    for run in all_live_runs:
        if run.dataset_id is None or run.run_id is None:
            continue
        runs_by_dataset[run.dataset_id].append(run)
    for run in all_archived_runs:
        if run.dataset_id is None or run.run_id is None:
            continue
        if run.dataset_id not in runs_by_dataset:
            logger.warning(
                "Run with ID %s has a dataset ID that is not in the datasets list",
                run.run_id,
            )
            continue
        runs_by_dataset[run.dataset_id].append(run)
    return runs_by_dataset


def _cleanup_storage_configs(one_week_ago: datetime.datetime) -> None:
    trying_to_delete_storage_configs = set[int]()
    deleted_storage_configs = set[int]()
    storage_configs = StorageConfig.list()
    for storage_config in storage_configs:
        if (
            "ubuntu-latest" in storage_config.name
            or "windows-latest" in storage_config.name
            or "macos-latest" in storage_config.name
        ) and storage_config.created_at < one_week_ago:
            trying_to_delete_storage_configs.add(storage_config.id)
            try:
                StorageConfig.delete_by_id(storage_config.id)
                deleted_storage_configs.add(storage_config.id)
            except (HirundoError, requests.HTTPError) as exc:
                logger.warning(
                    "Failed to delete storage config with ID %s: %s",
                    storage_config.id,
                    exc,
                )
    logger.info(
        "Deleted %s (%s) storage configs",
        deleted_storage_configs,
        len(deleted_storage_configs),
    )
    if trying_to_delete_storage_configs != deleted_storage_configs:
        logger.warning(
            "Tried to delete %s storage configs, but only deleted %s storage configs",
            trying_to_delete_storage_configs,
            deleted_storage_configs,
        )


def main() -> None:
    all_live_runs = QADataset.list_runs()
    all_archived_runs = QADataset.list_runs(archived=True)
    datasets = {
        dataset_entry.id: dataset_entry
        for dataset_entry in QADataset.list_datasets()
        if dataset_entry.id is not None
    }
    now = datetime.datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)

    runs_by_dataset = _collect_runs_by_dataset(
        datasets, all_live_runs, all_archived_runs
    )
    trying_to_delete_datasets = set[int]()
    deleted_datasets = set[int]()
    deleted_storage_configs = set[int]()
    deleted_git_repos = set[int]()
    archived_runs = set[str]()
    for dataset_id, dataset_runs in runs_by_dataset.items():
        dataset = datasets.get(dataset_id)

        if dataset and _should_delete_dataset(dataset.name, dataset_runs, one_week_ago):
            trying_to_delete_datasets.add(dataset_id)
            for run in dataset_runs:
                if run.deleted_at is not None:
                    continue
                try:
                    QADataset.archive_run_by_id(run.run_id)
                    archived_runs.add(run.run_id)
                except (HirundoError, requests.HTTPError) as exc:
                    logger.warning(
                        "Failed to archive run with ID %s: %s", run.run_id, exc
                    )

            deleted_datasets, deleted_storage_configs, deleted_git_repos = (
                _delete_dataset(
                    dataset_id,
                    dataset.storage_config,
                    deleted_datasets,
                    deleted_storage_configs,
                    deleted_git_repos,
                )
            )

    logger.info(
        "Deleted %s (%s) datasets, %s (%s) storage configs, %s (%s) git repos and archived %s (%s) runs",
        deleted_datasets,
        len(deleted_datasets),
        deleted_storage_configs,
        len(deleted_storage_configs),
        deleted_git_repos,
        len(deleted_git_repos),
        archived_runs,
        len(archived_runs),
    )
    if trying_to_delete_datasets != deleted_datasets:
        logger.warning(
            "Tried to delete %s datasets, but only deleted %s datasets",
            trying_to_delete_datasets,
            deleted_datasets,
        )

    _cleanup_storage_configs(one_week_ago)


if __name__ == "__main__":
    main()
