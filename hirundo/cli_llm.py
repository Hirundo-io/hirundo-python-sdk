from typing import Annotated

import typer

from hirundo._cli_common import hirundo_epilog, make_app, print_runs_table
from hirundo._llm_sources import HuggingFaceTransformersModelOutput, LlmSourcesOutput

llm_app = make_app("llm", "List the LLMs available for unlearning and evaluation.")


def _source_label(model_source: LlmSourcesOutput) -> str:
    if isinstance(model_source, HuggingFaceTransformersModelOutput):
        return model_source.model_name
    return model_source.local_path


@llm_app.command("list", epilog=hirundo_epilog)
def llm_list(
    organization_id: Annotated[
        int | None,
        typer.Option(
            "--organization-id",
            help="Only list LLMs that belong to this organization.",
        ),
    ] = None,
):
    """
    List LLMs registered in Hirundo.

    Use the ID column as --model-id for eval runs or as the model ID argument
    for unlearning runs.
    """
    from hirundo.unlearning_llm import LlmModel

    llm_models = LlmModel.list(organization_id=organization_id)
    print_runs_table(
        "LLMs:",
        ("ID", "Name", "Source Type", "Source", "Created At"),
        [
            (
                str(llm_model.id),
                llm_model.model_name,
                llm_model.model_source.type.value,
                _source_label(llm_model.model_source),
                llm_model.created_at.isoformat(),
            )
            for llm_model in llm_models
        ],
    )
