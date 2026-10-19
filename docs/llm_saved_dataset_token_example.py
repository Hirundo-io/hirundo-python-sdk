"""Save an existing Hugging Face credential in Hirundo and reuse its ID."""

import os

from hirundo import (
    HuggingFaceToken,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmRunInfo,
    LlmUnlearningRun,
)
from hirundo.unlearning_llm import CustomBehavior, CustomUtility, HuggingFaceDataset

# Supply an existing Hugging Face read token outside this script.
huggingface_token = os.environ["HF_TOKEN"]
saved_token = HuggingFaceToken(name="custom-dataset-downloads", token=huggingface_token)
saved_token_id = saved_token.create()

model = LlmModel(
    model_name="saved-token-unlearning-model",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
        token=huggingface_token,
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
                    token_id=saved_token_id,
                ),
                unbiased_dataset=HuggingFaceDataset(
                    hugging_face_dataset_name="your-org/retain-dataset",
                    token_id=saved_token_id,
                ),
            )
        ],
        target_utilities=[
            CustomUtility(
                dataset=HuggingFaceDataset(
                    hugging_face_dataset_name="your-org/utility-dataset",
                    token_id=saved_token_id,
                )
            )
        ],
    ),
)
