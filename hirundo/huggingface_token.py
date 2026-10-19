"""Save Hugging Face credentials in Hirundo for use through dataset token IDs."""

from pydantic import BaseModel, ConfigDict, Field

from hirundo._env import API_HOST
from hirundo._headers import get_headers
from hirundo._http import raise_for_status_with_reason, requests
from hirundo._timeouts import MODIFY_TIMEOUT


class HuggingFaceToken(BaseModel):
    """An existing Hugging Face credential to save in the Hirundo platform.

    Names must be unique within the selected organization. Omit
    ``organization_id`` to use the authenticated user's default organization.
    This registers a credential; it does not mint a Hugging Face access token.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    id: int | None = Field(default=None, exclude=True)
    name: str = Field(min_length=1, max_length=255)
    token: str = Field(min_length=1, repr=False)
    organization_id: int | None = None

    def create(self) -> int:
        """Save the credential and return its ID for dataset ``token_id`` fields."""
        response = requests.post(
            f"{API_HOST}/huggingface-token/",
            json=self.model_dump(mode="json"),
            headers=get_headers(),
            timeout=MODIFY_TIMEOUT,
        )
        raise_for_status_with_reason(response)
        token_id = response.json()["id"]
        self.id = token_id
        return token_id
