import logging
import os

import pytest
from hirundo import (
    BBQBiasType,
    BiasRunInfo,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmUnlearningRun,
)
from tests.testing_utils import get_unique_id
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


def test_unlearn_llm_behavior():
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-Qwen3-0.6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    BiasRunInfo(bias_type=BBQBiasType.RACE)
    assert llm_id is not None


@pytest.mark.skip(
    reason="SDK-97: backend preprocessing hangs on unlearning runs; re-enable once backend team resolves the issue"
)
def test_unlearn_llm_behavior_full():
    if os.getenv("FULL_TEST", "false") != "true":
        pytest.skip("FULL_TEST not enabled")
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-FULL-Qwen3-0.6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    run_info = BiasRunInfo(bias_type=BBQBiasType.RACE)
    assert llm_id is not None
    run_id = LlmUnlearningRun.launch(llm_id, run_info)
    new_adapter = llm.get_hf_pipeline_for_run(run_id)
    assert isinstance(new_adapter, Pipeline)
