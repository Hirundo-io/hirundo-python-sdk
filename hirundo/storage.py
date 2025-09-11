import typing
from pathlib import Path

import pydantic
from pydantic import BaseModel, model_validator
from pydantic_core import Url

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT, READ_TIMEOUT
from hirundo._urls import S3BucketUrl, StorageConfigName
from hirundo.dataset_enum import StorageTypes
from hirundo.git import GitRepo, GitRepoOut
from hirundo.logger import get_logger

logger = get_logger(__name__)

S3_PREFIX = "s3://"


class StorageS3Base(BaseModel):
    endpoint_url: typing.Optional[Url] = None
    bucket_url: S3BucketUrl
    region_name: str
    # ⬆️ We could restrict this, but if we're allowing custom endpoints then the validation may be wrong
    access_key_id: typing.Optional[str] = None

    def get_url(self, path: typing.Union[str, Path]) -> Url:
        """
        Get the full URL for a file in the S3 bucket

        Chains the bucket URL with the path, ensuring that the path is formatted correctly

        Args:
            path: The path to the file in the S3 bucket, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

        Returns:
            The full URL to the file in the S3 bucket, e.g. :file:`s3://my-bucket/my-file.txt` or :file:`s3://my-bucket/my-folder/my-file.txt`,
            where :file:`s3://my-bucket` is the bucket URL provided in the S3 storage config
        """
        return Url(
            f"{S3_PREFIX}{self.bucket_url.removeprefix(S3_PREFIX).removesuffix('/')}/{str(path).removeprefix('/')}"
        )


class StorageS3(StorageS3Base):
    secret_access_key: typing.Optional[str] = None


class StorageS3Out(StorageS3Base):
    pass


class StorageGCPBase(BaseModel):
    bucket_name: str
    project: str

    def get_url(self, path: typing.Union[str, Path]) -> Url:
        """
        Get the full URL for a file in the GCP bucket

        Chains the bucket URL with the path, ensuring that the path is formatted correctly

        Args:
            path: The path to the file in the GCP bucket, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

        Returns:
            The full URL to the file in the GCP bucket, e.g. :file:`gs://my-bucket/my-file.txt` or :file:`gs://my-bucket/my-folder/my-file.txt`,
            where :file:`my-bucket` is the bucket name provided in the GCP storage config
        """
        return Url(f"gs://{self.bucket_name}/{str(path).removeprefix('/')}")


class StorageGCP(StorageGCPBase):
    credentials_json: typing.Optional[dict] = None


class StorageGCPOut(StorageGCPBase):
    pass


# TODO: Azure storage config is coming soon
# class StorageAzure(BaseModel):
#     account_url: HttpUrl
#     container_name: str
#     tenant_id: str

#     def get_url(self, path: typing.Union[str, Path]) -> Url:
#         """
#         Get the full URL for a file in the Azure container

#         Chains the container URL with the path, ensuring that the path is formatted correctly

#         Args:
#             path: The path to the file in the Azure container, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

#         Returns:
#             The full URL to the file in the Azure container
#         """
#         return Url(f"{str(self.account_url)}/{self.container_name}/{str(path).removeprefix('/')}")
# class StorageAzureOut(BaseModel):
#     container: str
#     account_url: str


def get_git_repo_url(
    repo_url: typing.Union[str, Url], path: typing.Union[str, Path]
) -> Url:
    """
    Get the full URL for a file in the git repository

    Chains the repository URL with the path, ensuring that the path is formatted correctly

    Args:
        repo_url: The URL of the git repository, e.g. :file:`https://my-git-repository.com`
        path: The path to the file in the git repository, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

    Returns:
        The full URL to the file in the git repository, e.g. :file:`https://my-git-repository.com/my-file.txt` or :file:`https://my-git-repository.com/my-folder/my-file.txt`
    """
    if not isinstance(repo_url, Url):
        repo_url = Url(repo_url)
    return Url(
        f"{repo_url.scheme}{str(repo_url).removeprefix(repo_url.scheme)}/{str(path).removeprefix('/')}"
    )


class StorageGit(BaseModel):
    repo_id: typing.Optional[int] = None
    """
    The ID of the Git repository in the Hirundo system.
    Either :code:`repo_id` or :code:`repo` must be provided.
    """
    repo: typing.Optional[GitRepo] = None
    """
    The Git repository to link to.
    Either :code:`repo_id` or :code:`repo` must be provided.
    """
    branch: str
    """
    The branch of the Git repository to link to.
    """

    @model_validator(mode="after")
    def validate_repo(self):
        if self.repo_id is None and self.repo is None:
            raise ValueError("Either repo_id or repo must be provided")
        return self

    def get_url(self, path: typing.Union[str, Path]) -> Url:
        """
        Get the full URL for a file in the git repository

        Chains the repository URL with the path, ensuring that the path is formatted correctly

        Args:
            path: The path to the file in the git repository, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

        Returns:
            The full URL to the file in the git repository, e.g. :file:`https://my-git-repository.com/my-file.txt` or :file:`https://my-git-repository.com/my-folder/my-file.txt`,
            where :file:`https://my-git-repository.com` is the repository URL provided in the git storage config's git repo
        """
        if not self.repo:
            raise ValueError("Repo must be provided to use `get_url`")
        repo_url = self.repo.repository_url
        return get_git_repo_url(repo_url, path)


class StorageGitOut(BaseModel):
    repo: GitRepoOut
    branch: str

    def get_url(self, path: typing.Union[str, Path]) -> Url:
        """
        Get the full URL for a file in the git repository

        Chains the repository URL with the path, ensuring that the path is formatted correctly

        Args:
            path: The path to the file in the git repository, e.g. :file:`my-file.txt` or :file:`/my-folder/my-file.txt`

        Returns:
            The full URL to the file in the git repository, e.g. :file:`https://my-git-repository.com/my-file.txt` or :file:`https://my-git-repository.com/my-folder/my-file.txt`,
            where :file:`https://my-git-repository.com` is the repository URL provided in the git storage config's git repo
        """
        repo_url = self.repo.repository_url
        return get_git_repo_url(repo_url, path)


class StorageConfig(BaseModel):
    id: typing.Optional[int] = None
    """
    The ID of the :code:`StorageConfig` in the Hirundo system.
    """

    organization_id: typing.Optional[int] = None
    """
    The ID of the organization that the :code:`StorageConfig` belongs to.
    If not provided, it will be assigned to your default organization.
    """

    name: StorageConfigName
    """
    A name to identify the :code:`StorageConfig` in the Hirundo system.
    """
    type: typing.Optional[StorageTypes] = pydantic.Field(
        examples=[
            StorageTypes.S3,
            StorageTypes.GCP,
            # StorageTypes.AZURE,  TODO: Azure storage is coming soon
            StorageTypes.GIT,
        ]
    )
    """
    The type of the :code:`StorageConfig`.
    Supported types are:
    - :code:`S3`
    - :code:`GCP`
    - :code:`Azure` (coming soon)
    - :code:`Git`
    """
    s3: typing.Optional[StorageS3] = pydantic.Field(
        default=None,
        examples=[
            {
                "bucket_url": "s3://my-bucket",
                "region_name": "us-west-2",
                "access_key_id": "my-access-key",
                "secret_access_key": "REDACTED",
            },
            None,
            None,
            None,
        ],
    )
    """
    The Amazon Web Services (AWS) S3 storage config details.
    Use this if you want to link to an S3 bucket.
    """
    gcp: typing.Optional[StorageGCP] = pydantic.Field(
        default=None,
        examples=[
            None,
            {
                "bucket_name": "my-bucket",
                "project": "my-project",
                "credentials_json": {
                    "type": "service_account",
                    "project_id": "my-project",
                    "private_key_id": "my-key-id",
                    "private_key": "REDACTED",
                    "client_email": "my-service-account@my-project.iam.gserviceaccount.com",
                    "client_id": "my-id",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/my-service-account%40my-project.iam.gserviceaccount.com",
                    "universe_domain": "googleapis.com",
                },
            },
            None,
            None,
        ],
    )
    """
    The Google Cloud (GCP) Storage config details.
    Use this if you want to link to an GCS bucket.
    """
    azure: None = None
    # azure: typing.Optional[StorageAzure] = pydantic.Field(
    #     default=None,
    #     examples=[
    #         None,
    #         None,
    #         {
    #             "container": "my-container",
    #             "account_name": "my-account-name",
    #             "account_key": "my-account",
    #         },
    #         None,
    #     ],
    # )  TODO: Azure storage config is coming soon
    git: typing.Optional[StorageGit] = pydantic.Field(
        default=None,
        examples=[
            None,
            None,
            None,
            {
                "repo_id": "my-repo-id",
                "repo": {
                    "name": "test-dataset",
                    "repository_url": "https://github.com/Hirundo-io/test-dataset.git",
                },
                "branch": "main",
                "path": "/my-path/to/dataset",
            },
        ],
    )
    """
    The Git storage config details.
    Use this if you want to link to a Git repository.
    """

    @staticmethod
    def get_by_id(storage_config_id: int) -> "ResponseStorageConfig":
        """
        Retrieves a :code:`StorageConfig` instance from the server by its ID

        Args:
            storage_config_id: The ID of the :code:`StorageConfig` to retrieve
        """
        storage_config = requests.get(
            f"{API_HOST}/storage-config/{storage_config_id}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(storage_config)
        return ResponseStorageConfig(**storage_config.json())

    @staticmethod
    def get_by_name(name: str, storage_type: StorageTypes) -> "ResponseStorageConfig":
        """
        Retrieves a :code:`StorageConfig` instance from the server by its name

        Args:
            name: The name of the :code:`StorageConfig` to retrieve
            storage_type: The type of the :code:`StorageConfig` to retrieve

            Note: The type is required because the name is not unique across different storage types
        """
        storage_config = requests.get(
            f"{API_HOST}/storage-config/by-name/{name}?storage_type={storage_type.value}",
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(storage_config)
        return ResponseStorageConfig(**storage_config.json())

    @staticmethod
    def list(
        organization_id: typing.Optional[int] = None,
    ) -> list["ResponseStorageConfig"]:
        """
        Lists all the :code:`StorageConfig`'s created by user's default organization
        Note: The return type is :code:`list[dict]` and not :code:`list[StorageConfig]`

        Args:
            organization_id: The ID of the organization to list :code:`StorageConfig`'s for.
            If not provided, it will list :code:`StorageConfig`'s for the default organization.
        """
        storage_configs = requests.get(
            f"{API_HOST}/storage-config/",
            params={"storage_config_organization_id": organization_id},
            headers=get_headers(),
            timeout=READ_TIMEOUT,
        )
        raise_for_status_with_reason(storage_configs)
        return [ResponseStorageConfig(**si) for si in storage_configs.json()]

    @staticmethod
    def delete_by_id(storage_config_id) -> None:
        """
        Deletes a :code:`StorageConfig` instance from the server by its ID

        Args:
            storage_config_id: The ID of the :code:`StorageConfig` to delete
        """
        storage_config = requests.delete(
            f"{API_HOST}/storage-config/{storage_config_id}",
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(storage_config)
        logger.info("Deleted storage config with ID: %s", storage_config_id)

    def delete(self) -> None:
        """
        Deletes the :code:`StorageConfig` instance from the server
        """
        if not self.id:
            raise ValueError("No StorageConfig has been created")
        self.delete_by_id(self.id)

    def create(self, replace_if_exists: bool = False) -> int:
        """
        Create a :code:`StorageConfig` instance on the server

        Args:
            replace_if_exists: If a :code:`StorageConfig` with the same name and type already exists, replace it.
        """
        if self.git and self.git.repo:
            self.git.repo_id = self.git.repo.create(replace_if_exists=replace_if_exists)
        storage_config = requests.post(
            f"{API_HOST}/storage-config/",
            json={
                **self.model_dump(mode="json"),
                "replace_if_exists": replace_if_exists,
            },
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(storage_config)
        storage_config_id = storage_config.json()["id"]
        self.id = storage_config_id
        logger.info("Created storage config with ID: %s", storage_config_id)
        return storage_config_id

    @model_validator(mode="after")
    def validate_storage_type(self):
        if self.type != StorageTypes.LOCAL and (
            [self.s3, self.gcp, self.git].count(None) != 2
        ):
            raise ValueError("Exactly one of S3, GCP, or Git must be provided")
        if self.type == StorageTypes.S3 and self.s3 is None:
            raise ValueError("S3 storage details must be provided")
        elif self.type == StorageTypes.GCP and self.gcp is None:
            raise ValueError("GCP storage details must be provided")
        elif self.type == StorageTypes.GIT and self.git is None:
            raise ValueError("Git storage details must be provided")
        if not self.type and not any([self.s3, self.gcp, self.git]):
            raise ValueError("Storage type must be provided")
        elif not self.type:
            self.type = (
                StorageTypes.S3
                if self.s3 is not None
                else StorageTypes.GCP
                if self.gcp is not None
                else StorageTypes.GIT
                if self.git is not None
                else StorageTypes.LOCAL
            )
        return self


class ResponseStorageConfig(BaseModel):
    id: int
    name: StorageConfigName
    type: StorageTypes
    organization_name: str
    creator_name: str
    s3: typing.Optional[StorageS3Out]
    gcp: typing.Optional[StorageGCPOut]
    # azure: typing.Optional[StorageAzureOut]
    git: typing.Optional[StorageGitOut]
