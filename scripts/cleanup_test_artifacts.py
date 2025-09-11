import datetime
from collections import defaultdict
from datetime import timedelta, timezone
from typing import Union

import requests
from hirundo import GitRepo, QADataset, StorageConfig
from hirundo.dataset_qa import HirundoError, RunStatus
from hirundo.logger import get_logger
from hirundo.storage import ResponseStorageConfig

logger = get_logger(__name__)


def _delete_dataset(
    dataset_id: int,
    storage_config: Union[StorageConfig, ResponseStorageConfig, None],
) -> None:
    try:
        QADataset.delete_by_id(dataset_id)
    except (HirundoError, requests.HTTPError) as exc:
        logger.warning("Failed to delete dataset with ID %s: %s", dataset_id, exc)

    if storage_config and storage_config.id is not None:
        try:
            StorageConfig.delete_by_id(storage_config.id)
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
            except (HirundoError, requests.HTTPError) as exc:
                logger.warning(
                    "Failed to delete git repo with ID %s: %s", git_repo_id, exc
                )


def _should_delete_dataset(dataset_runs: list, expiry_date: datetime.datetime) -> bool:
    """Return ``True`` if the dataset should be deleted."""

    if not dataset_runs:
        return False

    all_runs_successful = all(run.status == RunStatus.SUCCESS for run in dataset_runs)
    if all_runs_successful:
        return True

    most_recent_run_time = max(run.created_at for run in dataset_runs)
    return most_recent_run_time <= expiry_date


def main() -> None:
    all_runs = QADataset.list_runs()
    datasets = {
        dataset_entry.id: dataset_entry
        for dataset_entry in QADataset.list_datasets()
        if dataset_entry.id is not None
    }
    now = datetime.datetime.now(timezone.utc)
    one_week_ago = now - timedelta(days=7)

    runs_by_dataset: defaultdict[int, list] = defaultdict(list)
    for run in all_runs:
        if run.dataset_id is None or run.run_id is None:
            continue
        runs_by_dataset[run.dataset_id].append(run)

    for dataset_id, dataset_runs in runs_by_dataset.items():
        dataset = datasets.get(dataset_id)
        if dataset is None or not dataset.name.startswith("TEST-"):
            continue

        if _should_delete_dataset(dataset_runs, one_week_ago):
            for run in dataset_runs:
                try:
                    QADataset.archive_run_by_id(run.run_id)
                except (HirundoError, requests.HTTPError) as exc:
                    logger.warning(
                        "Failed to archive run with ID %s: %s", run.run_id, exc
                    )

            _delete_dataset(dataset_id, dataset.storage_config)


if __name__ == "__main__":
    main()
