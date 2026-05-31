import io
import zipfile
from pathlib import Path

import pytest
from hirundo._dataframe import has_pandas, has_polars
from hirundo.unzip import download_and_extract_zip

SUSPECTS_CSV = "image_path,suspect_level\nimg_0.png,0.9\n"
SUSPECT_LEVEL_COUNTS_CSV = "suspect_level,count\n0.9,1\n"
WARNINGS_AND_ERRORS_CSV = "image_path,status\nimg_1.png,MISSING_IMAGE\n"


def _build_results_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mislabel_suspects.csv", SUSPECTS_CSV)
        archive.writestr("mislabel_suspect_level_counts.csv", SUSPECT_LEVEL_COUNTS_CSV)
        archive.writestr("warnings_and_errors.csv", WARNINGS_AND_ERRORS_CSV)
    return buffer.getvalue()


class _FakeStreamingResponse:
    """Minimal stand-in for a streaming `requests` response."""

    def __init__(self, content: bytes):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        yield self._content


@pytest.mark.skipif(
    not (has_pandas or has_polars),
    reason="Requires pandas or polars to materialize result DataFrames",
)
def test_download_and_extract_zip_populates_result_frames(monkeypatch, tmp_path):
    zip_bytes = _build_results_zip()
    monkeypatch.setattr(
        "hirundo.unzip.requests.get",
        lambda *args, **kwargs: _FakeStreamingResponse(zip_bytes),
    )
    # Redirect the cache dir to a temp path so the test never touches the real home.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    results = download_and_extract_zip("test-run-id", "https://example.com/results.zip")

    assert results.suspects is not None
    assert results.suspect_level_counts is not None
    assert results.warnings_and_errors is not None
    # `object_mislabel_suspects.csv` is absent from this ZIP, so it stays None.
    assert results.object_suspects is None
