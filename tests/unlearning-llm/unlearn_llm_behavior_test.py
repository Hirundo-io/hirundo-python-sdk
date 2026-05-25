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

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


def test_unlearn_llm_behavior():
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-Granite-4-micro-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="ibm-granite/granite-4.0-micro",
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
        # Verify the server-side unlearning job completes. Loading the full
        # ibm-granite/granite-4.0-micro (granitemoehybrid MoE) into the ~7 GB
        # RAM available on a GitHub Actions runner causes OOM. The pipeline-
        # loading path is already covered by the mock-based tests in
        # llm_pipeline_transformers_test.py.
        result = LlmUnlearningRun.check_run_by_id(run_id)
        assert result is not None
