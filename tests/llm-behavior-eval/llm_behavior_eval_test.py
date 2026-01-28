import logging
import os

from hirundo import (
    BiasType,
    EvalRunInfo,
    HuggingFaceTransformersModel,
    LlmBehaviorEval,
    LlmModel,
    ModelOrRun,
    PresetType,
)
from tests.testing_utils import get_unique_id

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


def test_llm_behavior_eval():
    llm = LlmModel(
        model_name=f"TEST-LLM-BEHAVIOR-EVAL-Granite-4-micro-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="ibm-granite/granite-4.0-micro",
        ),
    )
    llm_id = llm.create()
    run_info = EvalRunInfo(
        name=f"TEST-LLM-BEHAVIOR-EVAL-RUN-{unique_id}",
        model_id=llm_id,
        preset_type=PresetType.BBQ_BIAS,
        bias_type=BiasType.ALL,
    )
    assert llm_id is not None
    if os.getenv("FULL_TEST", "false") == "true":
        run_id = LlmBehaviorEval.launch_eval_run(ModelOrRun.MODEL, run_info)
        assert run_id is not None
        results = LlmBehaviorEval.check_run_by_id(run_id)
        assert results is not None
        assert results.cached_zip_path is not None
