"""Editable CUSTOM sources initialized from Security, with explicit credentials."""

import os

from hirundo import HuggingFaceTransformersModel, LlmModel, LlmRunInfo, LlmUnlearningRun
from hirundo.unlearning_llm import (
    CustomBehavior,
    CustomDataset,
    CustomUtility,
    HuggingFaceDataset,
)

# Set HF_TOKEN outside this script when the repositories require authentication.
huggingface_token = os.getenv("HF_TOKEN")

# Each resource can use a different credential; these defaults reuse HF_TOKEN.
model_token = huggingface_token
forget_token = huggingface_token
retain_token = huggingface_token
utility_token = huggingface_token

# Edit these independent collections; keep forget and retain nonempty.
# Initial forget/retain sources match the Security preset as of 2026-10-19.
FORGET_DATASETS: list[CustomDataset] = [
    HuggingFaceDataset(
        hugging_face_dataset_name=(
            "hirundo-io/spml-chatbot-prompt-injection-malicious-refusals"
        ),
        token=forget_token,
    ),
]
RETAIN_DATASETS: list[CustomDataset] = [
    HuggingFaceDataset(
        hugging_face_dataset_name="hirundo-io/iheval-benign-answers",
        token=retain_token,
    ),
]
# The platform already adds Security's built-in utilities: hirundo-io/ifbench,
# hirundo-io/scicode, and hirundo-io/mmlu-pro-multi-choice. Do not duplicate them.
ADDITIONAL_UTILITY_DATASETS: list[CustomDataset] = [
    # HuggingFaceDataset(
    #     hugging_face_dataset_name="your-org/extra-utility", token=utility_token
    # ),
]


def main() -> None:
    """Register the model and launch one run on a platform supporting HPLT-377."""
    model = LlmModel(
        model_name="custom-unlearning-model",
        model_source=HuggingFaceTransformersModel(
            model_name="nvidia/Nemotron-Flash-1B",
            token=model_token,
        ),
    )
    model_id = model.create()
    LlmUnlearningRun.launch(
        model_id,
        LlmRunInfo(
            target_behaviors=[
                CustomBehavior(
                    biased_dataset=FORGET_DATASETS,
                    unbiased_dataset=RETAIN_DATASETS,
                )
            ],
            target_utilities=[
                CustomUtility(dataset=dataset)
                for dataset in ADDITIONAL_UTILITY_DATASETS
            ],
        ),
    )


if __name__ == "__main__":
    main()
