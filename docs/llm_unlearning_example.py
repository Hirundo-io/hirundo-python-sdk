"""Examples for docs/index.rst literalinclude blocks."""

from hirundo import (
    BiasBehavior,
    HuggingFaceTransformersModel,
    LlmRunInfo,
    LlmModel,
    LlmUnlearningRun,
)

llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()
run_id = LlmUnlearningRun.launch(
    llm_id,
    LlmRunInfo(target_behaviors=[BiasBehavior()]),
)
result = LlmUnlearningRun.check_run(run_id)
new_adapter = llm.get_hf_pipeline_for_run(run_id)
