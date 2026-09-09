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

# Refusal presets do not use a bias type. They can evaluate a base model or an
# unlearning run by selecting ModelOrRun.MODEL or ModelOrRun.RUN.
refusal_run_id = LlmBehaviorEval.launch_eval_run(
    ModelOrRun.MODEL,
    EvalRunInfo(
        name="Nemotron XSTest refusal eval",
        model_id=llm_id,
        preset_type=PresetType.XSTEST,
    ),
)

or_bench_run_id = LlmBehaviorEval.launch_eval_run(
    ModelOrRun.RUN,
    EvalRunInfo(
        name="ORBench refusal eval",
        source_run_id="unlearning-run-id",
        preset_type=PresetType.OR_BENCH,
    ),
)
