"""Examples for docs/index.rst literalinclude blocks."""

from hirundo import (
    BBQBiasType,
    EvalRunInfo,
    HuggingFaceTransformersModel,
    LlmBehaviorEval,
    LlmModel,
    ModelOrRun,
    PresetType,
)

llm = LlmModel(
    model_name="Nemotron-Flash-1B",
    model_source=HuggingFaceTransformersModel(
        model_name="nvidia/Nemotron-Flash-1B",
    ),
)
llm_id = llm.create()

run_id = LlmBehaviorEval.launch_eval_run(
    ModelOrRun.MODEL,
    EvalRunInfo(
        name="Nemotron BBQ bias eval",
        model_id=llm_id,
        preset_type=PresetType.BBQ_BIAS,
        bias_type=BBQBiasType.ALL,
    ),
)

results = LlmBehaviorEval.check_run_by_id(run_id)
print(results.summary_brief)
