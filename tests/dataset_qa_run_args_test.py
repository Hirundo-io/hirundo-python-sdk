import pytest
from hirundo.dataset_qa import (
    ClassificationRunArgs,
    ObjectDetectionRunArgs,
    QADataset,
)
from pydantic import JsonValue


@pytest.mark.parametrize(
    ("wire_args", "expected"),
    [
        (
            {"img_size": [128, 96], "upsample": True},
            ClassificationRunArgs(img_size=(128, 96), upsample=True),
        ),
        (
            {
                "img_size": [64, 80],
                "upsample": True,
                "min_abs_bbox_size": 8,
                "min_abs_bbox_area": 64,
                "min_rel_bbox_size": 0.1,
                "min_rel_bbox_area": 0.2,
                "crop_ratio": 0.8,
                "add_mask_channel": True,
            },
            ObjectDetectionRunArgs(
                img_size=(64, 80),
                upsample=True,
                min_abs_bbox_size=8,
                min_abs_bbox_area=64,
                min_rel_bbox_size=0.1,
                min_rel_bbox_area=0.2,
                crop_ratio=0.8,
                add_mask_channel=True,
            ),
        ),
        ({"img_size": None}, ClassificationRunArgs(img_size=None)),
        (
            {"img_size": [48, 72], "image_size": [128, 96]},
            ClassificationRunArgs(img_size=(48, 72)),
        ),
        ({}, ClassificationRunArgs()),
        (None, None),
    ],
)
def test_list_runs_preserves_server_run_args(
    monkeypatch: pytest.MonkeyPatch,
    wire_args: JsonValue,
    expected: ClassificationRunArgs | None,
) -> None:
    payload: dict[str, JsonValue] = {
        "id": 1,
        "name": "run",
        "dataset_id": 123,
        "run_id": "run-123",
        "status": "PENDING",
        "approved": False,
        "created_at": "2026-06-22T14:20:31.663Z",
        "run_args": wire_args,
    }

    class Response:
        status_code = 200

        def json(self) -> list[dict[str, JsonValue]]:
            return [payload]

        def raise_for_status(self) -> None:
            return None

    def fake_get(
        url: str,
        *,
        params: dict[str, JsonValue],
        headers: dict[str, str],
        timeout: float,
    ) -> Response:
        return Response()

    monkeypatch.setattr("hirundo.dataset_qa.requests.get", fake_get)

    result = QADataset.list_runs()[0]

    assert result.run_args == expected
    if expected is not None:
        assert type(result.run_args) is type(expected)
        assert result.model_dump(mode="json")["run_args"] == expected.model_dump(
            mode="json"
        )
    assert payload["run_args"] == wire_args


@pytest.mark.parametrize("model", [ClassificationRunArgs, ObjectDetectionRunArgs])
def test_run_args_expose_only_server_image_size_field(
    model: type[ClassificationRunArgs],
) -> None:
    properties = model.model_json_schema()["properties"]
    assert "img_size" in properties
    assert "image_size" not in properties
    args = model.model_validate({"image_size": [48, 72]})
    assert args.img_size == (224, 224)
    assert "image_size" not in args.model_dump()
