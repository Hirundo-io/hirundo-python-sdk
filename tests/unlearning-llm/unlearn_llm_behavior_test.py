import logging
import os

import pytest
from hirundo import (
    BiasBehavior,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmRunInfo,
    LlmUnlearningRun,
)
from tests.testing_utils import get_unique_id
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


def test_unlearn_llm_behavior():
    if os.getenv("FULL_TEST", "false") != "true":
        pytest.skip("FULL_TEST not enabled")

    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-Qwen3-0.6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()])
    assert llm_id is not None
    run_id = None
    try:
        run_id = LlmUnlearningRun.launch(llm_id, run_info)
        new_adapter = llm.get_hf_pipeline_for_run(run_id)
        assert isinstance(new_adapter, Pipeline)
    finally:
        if run_id is not None:
            LlmUnlearningRun.archive(run_id)
        llm.delete()
