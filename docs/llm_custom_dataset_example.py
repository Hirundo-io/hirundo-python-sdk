"""Explicitly share an optional Hugging Face credential across custom resources."""

import os

from hirundo import HuggingFaceTransformersModel, LlmModel, LlmRunInfo, LlmUnlearningRun
from hirundo.unlearning_llm import CustomBehavior, CustomUtility, HuggingFaceDataset

# Set HF_TOKEN outside this script when the repositories require authentication.
huggingface_token = os.getenv("HF_TOKEN")

# Each resource can use a different credential; these defaults reuse HF_TOKEN.
model_token = huggingface_token
forget_token = huggingface_token
retain_token = huggingface_token
utility_token = huggingface_token

model = LlmModel(
    model_name="custom-unlearning-model",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
        token=model_token,
    ),
)
model_id = model.create()
run_id = LlmUnlearningRun.launch(
    model_id,
    LlmRunInfo(
        target_behaviors=[
            CustomBehavior(
                biased_dataset=HuggingFaceDataset(
                    hugging_face_dataset_name="your-org/forget-dataset",
                    token=forget_token,
                ),
                unbiased_dataset=HuggingFaceDataset(
                    hugging_face_dataset_name="your-org/retain-dataset",
                    token=retain_token,
                ),
            )
        ],
        target_utilities=[
            CustomUtility(
                dataset=HuggingFaceDataset(
                    hugging_face_dataset_name="your-org/utility-dataset",
                    token=utility_token,
                )
            )
        ],
    ),
)
