from hirundo import unzip


def test_download_request_converts_dataset_qa_file_url_to_local_download_query(
    monkeypatch,
) -> None:
    monkeypatch.setattr(unzip, "API_HOST", "http://localhost:8000")
    monkeypatch.setattr(
        unzip,
        "_get_auth_headers",
        lambda: {"Authorization": "Bearer test-token"},
    )

    zip_url, headers = unzip._download_request(
        "file:///datasets/results/statlog local/run-1/results.zip",
        "dataset-qa",
    )

    assert zip_url == (
        "http://localhost:8000/dataset-qa/run/local-download/"
        "?path=/datasets/results/statlog%20local/run-1/results.zip"
    )
    assert headers == {"Authorization": "Bearer test-token"}


def test_download_request_leaves_remote_url_unchanged() -> None:
    zip_url, headers = unzip._download_request(
        "https://storage.example.com/results.zip",
        "dataset-qa",
    )

    assert zip_url == "https://storage.example.com/results.zip"
    assert headers is None
