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
    # Qwen3-0.6B is a small dense (non-MoE) model (~751M params, ~3 GB fp32 /
    # ~1.5 GB fp16) chosen so the full integration test — server-side
    # unlearning plus local pipeline loading — fits in the ~7 GB RAM of the
    # GitHub Actions ubuntu-latest runner.
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-Qwen3-0_6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    run_info = BiasRunInfo(
        bias_type=BBQBiasType.ALL,
    )
    assert llm_id is not None
    if os.getenv("FULL_TEST", "false") == "true":
        run_id = LlmUnlearningRun.launch(
            llm_id,
            run_info,
        )
        new_adapter = llm.get_hf_pipeline_for_run(run_id, device_map="auto")
        assert isinstance(new_adapter, Pipeline)
