"""Examples for docs/index.rst literalinclude blocks."""

from hirundo import ExternalEval, ExternalEvalRunInfo, ModelOrRun

catalog = ExternalEval.get_catalog()
task_id = catalog.benchmarks[0].tasks[0].id

run_id = ExternalEval.launch_eval_run(
    ModelOrRun.MODEL,
    ExternalEvalRunInfo(
        name="Inspect evaluation",
        model_id=123,
        task_ids=[task_id],
        sample_limit=10,
    ),
)

# Use LlmBehaviorEval's unified lifecycle APIs to poll and download results.
print(run_id)
