import logging

from hirundo import (
    BiasRunInfo,
    BiasType,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmUnlearningRun,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


def test_unlearn_llm_behavior():
    llm_id = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-SmolLM2-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="HuggingFaceTB/SmolLM2-135M",
        ),
    ).create()
    run_info = BiasRunInfo(
        bias_type=BiasType.ALL,
    )
    LlmUnlearningRun.launch(
        llm_id,
        run_info,
    )
