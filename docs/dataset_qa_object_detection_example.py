"""Examples for docs/index.rst literalinclude blocks."""

from hirundo import (
    GitRepo,
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGit,
    StorageTypes,
)

git_storage = StorageGit(
    repo=GitRepo(
        name="BDD-100k-validation-dataset",
        repository_url=(
            "https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only"
        ),
    ),
    branch="main",
)

test_dataset = QADataset(
    name="TEST-HuggingFace-BDD-100k-validation-OD-validation-dataset",
    labeling_type=LabelingType.OBJECT_DETECTION,
    storage_config=StorageConfig(
        name="BDD-100k-validation-dataset",
        type=StorageTypes.GIT,
        git=git_storage,
    ),
    data_root_url=git_storage.get_url(path="/BDD100K Val from Hirundo.zip/bdd100k"),
    labeling_info=HirundoCSV(
        csv_url=git_storage.get_url(
            path="/BDD100K Val from Hirundo.zip/bdd100k/bdd100k.csv"
        ),
    ),
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
