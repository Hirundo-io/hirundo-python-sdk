"""Examples for docs/index.rst literalinclude blocks."""

# -- llm-unlearning-example-start
from hirundo import (
    BiasRunInfo,
    BiasType,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmUnlearningRun,
)

llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()
run_id = LlmUnlearningRun.launch(
    llm_id,
    BiasRunInfo(bias_type=BiasType.ALL),
)
result = LlmUnlearningRun.check_run(run_id)
new_adapter = llm.get_hf_pipeline_for_run(run_id)
# -- llm-unlearning-example-end

# -- dataset-qa-example-start
import json
import os

from hirundo import (
    HirundoCSV,
    LabelingType,
    QADataset,
    StorageConfig,
    StorageGCP,
    StorageTypes,
)

gcp_bucket = StorageGCP(
    bucket_name="cifar100bucket",
    project="Hirundo-global",
    credentials_json=json.loads(os.environ["GCP_CREDENTIALS"]),
)

cifar100_classes = ["apple", "bicycle", "cloud"]

test_dataset = QADataset(
    name="TEST-GCP cifar 100 classification dataset",
    labeling_type=LabelingType.SINGLE_LABEL_CLASSIFICATION,
    storage_config=StorageConfig(
        name="cifar100bucket",
        type=StorageTypes.GCP,
        gcp=gcp_bucket,
    ),
    data_root_url=gcp_bucket.get_url(path="/pytorch-cifar/data"),
    labeling_info=HirundoCSV(
        csv_url=gcp_bucket.get_url(path="/pytorch-cifar/data/cifar100.csv"),
    ),
    classes=cifar100_classes,
)

test_dataset.run_qa()
results = test_dataset.check_run()
print(results)
# -- dataset-qa-example-end
