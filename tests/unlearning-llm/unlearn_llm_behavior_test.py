import logging
import os

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
    run_info = BiasRunInfo(
        bias_type=BBQBiasType.RACE,
    )
    assert llm_id is not None
    # SDK-97: backend deadlocks in Map(num_proc=6) during data preprocessing
    # regardless of bias type — skipping full run until backend fixes num_proc
    # if os.getenv("FULL_TEST", "false") == "true":
    #     run_id = LlmUnlearningRun.launch(
    #         llm_id,
    #         run_info,
    #     )
    #     new_adapter = llm.get_hf_pipeline_for_run(run_id)
    #     assert isinstance(new_adapter, Pipeline)
