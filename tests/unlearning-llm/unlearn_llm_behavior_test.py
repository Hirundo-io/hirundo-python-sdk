import logging

from hirundo import (
    BBQBiasType,
    BiasRunInfo,
    HuggingFaceTransformersModel,
    LlmModel,
)
from tests.testing_utils import get_unique_id

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
    # Instantiate run_info to validate the BiasRunInfo constructor is importable
    # and BBQBiasType works; the actual backend run is skipped (SDK-97: backend
    # runs hang indefinitely during data preprocessing).
    BiasRunInfo(bias_type=BBQBiasType.RACE)
    assert llm_id is not None
    # TODO SDK-97: re-enable once backend team resolves the preprocessing hang
    # import os
    # from hirundo import LlmUnlearningRun
    # from transformers.pipelines.base import Pipeline
    # if os.getenv("FULL_TEST", "false") == "true":
    #     run_id = LlmUnlearningRun.launch(llm_id, run_info)
    #     new_adapter = llm.get_hf_pipeline_for_run(run_id)
    #     assert isinstance(new_adapter, Pipeline)
