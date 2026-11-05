"""Response bodies shaped like the generated OpenAPI response models."""

from pydantic import JsonValue

JsonObject = dict[str, JsonValue]


def create_response_payload(resource_id: JsonValue) -> JsonObject:
    """Build a `CreateModel` body returned by the API's create endpoints.

    Args:
        resource_id: Value for the `id` field.

    Returns:
        A JSON object with every field `CreateModel` requires.
    """
    return {
        "id": resource_id,
        "message": "Created",
        "created_at": "2026-11-05T12:00:00Z",
        "updated_at": "2026-11-05T12:00:00Z",
    }


def launch_response_payload(run_id: JsonValue) -> JsonObject:
    """Build a `LaunchRunResponse` body returned by the run launch endpoints.

    Args:
        run_id: Value for the `run_id` field.

    Returns:
        A JSON object with every field `LaunchRunResponse` requires.
    """
    return {"message": "Run launched", "run_id": run_id}
