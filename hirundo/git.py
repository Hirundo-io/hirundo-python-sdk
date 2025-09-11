import datetime
import re
import typing

import pydantic
from pydantic import BaseModel, field_validator
from pydantic_core import Url

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo._urls import RepoUrl
from hirundo.logger import get_logger

logger = get_logger(__name__)


class GitPlainAuth(BaseModel):
    username: str
    """
    The username for the Git repository
    """
    password: str
    """
    The password for the Git repository
    """


class GitSSHAuth(BaseModel):
    ssh_key: str
    """
    The SSH key for the Git repository
    """
    ssh_password: typing.Optional[str]
    """
    The password for the SSH key for the Git repository.
    """


class GitRepo(BaseModel):
    id: typing.Optional[int] = None
    """
    The ID of the Git repository.
    """

    name: str
    """
    A name to identify the Git repository in the Hirundo system.
    """
    repository_url: typing.Union[str, RepoUrl]
    """
    The URL of the Git repository, it should start with `ssh://` or `https://` or be in the form `user@host:path`.
    If it is in the form `user@host:path`, it will be rewritten to `ssh://user@host/path`.
    """
    organization_id: typing.Optional[int] = None
    """
    The ID of the organization that the Git repository belongs to.
    If not provided, it will be assigned to your default organization.
    """

    plain_auth: typing.Optional[GitPlainAuth] = pydantic.Field(
        default=None, examples=[None, {"username": "ben", "password": "password"}]
    )
    """
    The plain authentication details for the Git repository.
    Use this if using a special user with a username and password for authentication.
    """
    ssh_auth: typing.Optional[GitSSHAuth] = pydantic.Field(
        default=None,
        examples=[
            {
                "ssh_key": "SOME_PRIVATE_SSH_KEY",
                "ssh_password": "SOME_SSH_KEY_PASSWORD",
            },
            None,
        ],
    )
    """
    The SSH authentication details for the Git repository.
    Use this if using an SSH key for authentication.
    Optionally, you can provide a password for the SSH key.
    """

    @field_validator("repository_url", mode="before", check_fields=True)
    @classmethod
    def check_valid_repository_url(cls, repository_url: typing.Union[str, RepoUrl]):
        # Check if the URL has the `@` and `:` pattern with a non-numeric section before the next slash
        match = re.match("([^@]+@[^:]+):([^0-9/][^/]*)/(.+)", str(repository_url))
        if match:
            user_host = match.group(1)
            path = match.group(2) + "/" + match.group(3)
            rewritten_url = Url(f"ssh://{user_host}/{path}")
            # Check if the URL already has a protocol
            url_scheme = rewritten_url.scheme
            logger.info(
                "Modified Git repo to replace %s@%s:%s/%s with %s",
                url_scheme,
                match.group(1),
                match.group(2),
                match.group(3),
                rewritten_url,
            )
            return rewritten_url
        if not str(repository_url).startswith("ssh://") and not str(
            repository_url
        ).startswith("https://"):
            raise ValueError("Repository URL must start with 'ssh://' or 'https://'")
        if not isinstance(repository_url, Url):
            repository_url = Url(repository_url)
        return repository_url

    def create(self, replace_if_exists: bool = False) -> int:
        """
        Create a Git repository in the Hirundo system.

        Args:
            replace_if_exists: If a Git repository with the same name already exists, replace it.
        """
        git_repo = requests.post(
            f"{API_HOST}/git-repo/",
            json={
                **self.model_dump(mode="json"),
                "replace_if_exists": replace_if_exists,
            },
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(git_repo)
        git_repo_id = git_repo.json()["id"]
        self.id = git_repo_id
        return git_repo_id

    @staticmethod
    def get_by_id(git_repo_id: int) -> "GitRepoOut":
        """
        Retrieves a `GitRepo` instance from the server by its ID

        Args:
            git_repo_id: The ID of the `GitRepo` to retrieve
        """
        git_repo = requests.get(
            f"{API_HOST}/git-repo/{git_repo_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(git_repo)
        return GitRepoOut(**git_repo.json())

    @staticmethod
    def get_by_name(
        name: str,
    ) -> "GitRepoOut":
        """
        Retrieves a `GitRepo` instance from the server by its name

        Args:
            name: The name of the `GitRepo` to retrieve
        """
        git_repo = requests.get(
            f"{API_HOST}/git-repo/by-name/{name}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(git_repo)
        return GitRepoOut(**git_repo.json())

    @staticmethod
    def list() -> list["GitRepoOut"]:
        """
        List all Git repositories in the Hirundo system.
        """
        git_repos = requests.get(
            f"{API_HOST}/git-repo/",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(git_repos)
        git_repo_json = git_repos.json()
        return [
            GitRepoOut(
                **git_repo,
            )
            for git_repo in git_repo_json
        ]

    @staticmethod
    def delete_by_id(git_repo_id: int):
        """
        Delete a Git repository by its ID.

        Args:
            git_repo_id: The ID of the Git repository to delete
        """
        git_repo = requests.delete(
            f"{API_HOST}/git-repo/{git_repo_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(git_repo)

    def delete(self):
        """
        Delete the Git repository created by this instance.
        """
        if not self.id:
            raise ValueError("No GitRepo has been created")
        GitRepo.delete_by_id(self.id)


class GitRepoOut(BaseModel):
    id: int
    name: str
    repository_url: RepoUrl

    created_at: datetime.datetime
    updated_at: datetime.datetime
