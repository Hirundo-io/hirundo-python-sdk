import os

from hirundo import GitRepo


def test_git_repository_create_list_delete_sequence() -> None:
    unique_id = os.environ.get("UNIQUE_ID", "local-recording")
    repository = GitRepo(
        name=f"sdk-http-recording-{unique_id}",
        repository_url="https://github.com/Hirundo-io/hirundo-python-sdk",
    )

    repository_id = repository.create(replace_if_exists=True)
    try:
        listed_repositories = GitRepo.list()

        assert any(
            item.id == repository_id and item.name == repository.name
            for item in listed_repositories
        )
    finally:
        repository.delete()
