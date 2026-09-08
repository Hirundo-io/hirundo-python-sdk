"""Examples for docs/index.rst literalinclude blocks."""

from hirundo import ExternalEval, ExternalEvalRunInfo, ModelOrRun

catalog = ExternalEval.get_catalog()
task_id = catalog.benchmarks[0].tasks[0].id

launch = ExternalEval.launch_eval_run(
    ModelOrRun.MODEL,
    ExternalEvalRunInfo(
        name="Inspect evaluation",
        model_id=123,
        task_ids=[task_id],
        sample_limit=10,
    ),
)

# Poll and download the framework-neutral Inspect archive.
result = ExternalEval.check_run_by_id(launch.run_id)
print(result.cached_zip_path)
