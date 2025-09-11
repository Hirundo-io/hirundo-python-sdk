import os
import typing
from collections import defaultdict
from contextlib import contextmanager

import requests
from hirundo import (
    GitRepo,
    QADataset,
    RunArgs,
    StorageConfig,
)
from hirundo.dataset_qa import RunStatus
from hirundo.logger import get_logger

logger = get_logger(__name__)


def get_unique_id():
    return (
        os.getenv("UNIQUE_ID", "").replace(".", "-").replace("/", "-").replace("+", "-")
    )


def cleanup_conflict_by_unique_id(unique_id: typing.Optional[str]):
    if not unique_id:
        return
    runs = QADataset.list_runs()
    conflicting_run_ids = [run.run_id for run in runs if unique_id in run.name]
    datasets = QADataset.list_datasets()
    conflicting_dataset_ids = [
        dataset.id for dataset in datasets if unique_id in dataset.name
    ]
    conflicting_git_repo_ids = [
        git_repo.id for git_repo in GitRepo.list() if unique_id in git_repo.name
    ]
    conflicting_storage_config_ids = [
        storage_config.id
        for storage_config in StorageConfig.list()
        if unique_id in storage_config.name and storage_config.id is not None
    ]
    for conflicting_run_id in conflicting_run_ids:
        try:
            QADataset.archive_run_by_id(conflicting_run_id)
        except Exception as e:
            logger.warning(
                "Failed to archive run with ID %s and exception %s",
                conflicting_run_id,
                e,
            )
    for conflicting_dataset_id in conflicting_dataset_ids:
        try:
            QADataset.delete_by_id(conflicting_dataset_id)
        except Exception as e:
            logger.warning(
                "Failed to delete dataset with ID %s and exception %s",
                conflicting_dataset_id,
                e,
            )
    for conflicting_storage_config_id in conflicting_storage_config_ids:
        try:
            StorageConfig.delete_by_id(conflicting_storage_config_id)
        except Exception as e:
            logger.warning(
                "Failed to delete storage config with ID %s and exception %s",
                conflicting_storage_config_id,
                e,
            )
    for conflicting_git_repo_id in conflicting_git_repo_ids:
        try:
            GitRepo.delete_by_id(conflicting_git_repo_id)
        except Exception as e:
            logger.warning(
                "Failed to delete git repo with ID %s and exception %s",
                conflicting_git_repo_id,
                e,
            )


@contextmanager
def _handle_not_found_error(dataset: QADataset):
    try:
        yield
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.info(
                "QA dataset with name %s not found, skipping cleanup",
                dataset.name,
            )
            return
        else:
            raise e


def _get_runs_by_dataset():
    runs = QADataset.list_runs()
    runs_by_dataset = defaultdict(list)
    for run in runs:
        if run.dataset_id is not None and run.run_id is not None:
            runs_by_dataset[run.dataset_id].append(run.run_id)
    return runs_by_dataset


def cleanup(test_dataset: QADataset):
    logger.info("Started cleanup")
    with _handle_not_found_error(test_dataset):
        dataset = QADataset.get_by_name(test_dataset.name)
        storage_config_id = (
            dataset.storage_config.id if dataset.storage_config is not None else None
        )
        runs_by_dataset = _get_runs_by_dataset()
        if dataset.id is not None:
            logger.debug(
                "Found QA dataset with the same name, deleting it",
            )
            logger.debug(
                "Note: If I am not the owner, I will not be able to delete them"
            )
            try:
                if dataset.id in runs_by_dataset:
                    for run_id in runs_by_dataset[dataset.id]:
                        logger.debug("Archiving QA dataset with run ID %s", run_id)
                        QADataset.archive_run_by_id(run_id)
                QADataset.delete_by_id(dataset.id)
            except Exception as e:
                logger.warning(
                    "Unable to delete QA dataset with ID %s and exception %s",
                    dataset.id,
                    e,
                )
        if dataset.storage_config is not None and dataset.storage_config_id is not None:
            storage_config = dataset.storage_config
            storage_config_id = dataset.storage_config_id
            logger.debug(
                "Found storage config with ID %s, deleting it", storage_config_id
            )
            logger.debug("Note: If I am not the owner, I will not be able to delete it")
            try:
                StorageConfig.delete_by_id(storage_config_id)
            except Exception as e:
                logger.warning(
                    "Unable to delete storage config with ID %s and exception %s",
                    storage_config_id,
                    e,
                )
            if (
                storage_config.git is not None
                and storage_config.git.repo is not None
                and storage_config.git.repo.id is not None
            ):
                git_repo_id = storage_config.git.repo.id
                logger.debug("Found git repo with ID %s, deleting it", git_repo_id)
                logger.debug(
                    "Note: If I am not the owner, I will not be able to delete it"
                )
                try:
                    GitRepo.delete_by_id(git_repo_id)
                except Exception as e:
                    logger.warning(
                        "Unable to delete git repo with ID %s and exception %s",
                        git_repo_id,
                        e,
                    )
        # ⬆️ Delete all datasets and storage configs from the server for the given user's default organization
        test_dataset.clean_ids()
        # ⬆️ Reset `dataset_id`, `storage_config_id`, and `run_id` values on `test_dataset` to default value of `None`
        # This prevents errors due to ID links to deleted datasets, storage configs and runs
        logger.info("Finished cleanup")


def dataset_qa_sync_test(
    test_dataset: QADataset,
    alternative_env: typing.Optional[str] = None,
    sanity=False,
    run_args: typing.Optional[RunArgs] = None,
):
    logger.info("Sync: Finished cleanup")
    if (os.getenv("FULL_TEST", "false") == "true" and sanity) or (
        alternative_env and os.getenv(alternative_env, "false") == "true"
    ):
        run_id = test_dataset.run_qa(replace_dataset_if_exists=True, run_args=run_args)
        logger.info("Sync: Started dataset QA run with run ID %s", run_id)
        logger.info("Sync: Checking run progress")
        result = test_dataset.check_run(stop_on_manual_approval=True)
        logger.info("Sync: Results %s", result)
        return result
    else:
        test_dataset.create(replace_if_exists=True)
        logger.info("Sync: Created dataset %s", test_dataset.name)
        return None


async def dataset_qa_async_test(
    test_dataset: QADataset,
    env: str,
    run_args: typing.Optional[RunArgs] = None,
):
    logger.info("Async: Finished cleanup")
    if os.getenv(env, "false") == "true":
        run_id = test_dataset.run_qa(replace_dataset_if_exists=True, run_args=run_args)
        logger.info("Async: Started dataset QA run with run ID %s", run_id)
        events_generator = test_dataset.acheck_run()
        logger.info("Async: Checking run progress")
        last_event = {}
        async for last_event in events_generator:
            assert last_event is not None
            logger.info("Async: Run event %s", last_event)
            if last_event["state"] == RunStatus.AWAITING_MANUAL_APPROVAL:
                # Currently we require manual approval
                break
        logger.info("Async: Results %s", last_event["result"])
        return last_event["result"]
    else:
        test_dataset.create(replace_if_exists=True)
        logger.info("Async: Created dataset %s", test_dataset.name)
        return None
