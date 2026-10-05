import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from hirundo import (
    BiasBehavior,
    HuggingFaceTransformersModel,
    LlmModel,
    LlmRunInfo,
    LlmUnlearningRun,
)
from tests.testing_utils import get_unique_id
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

unique_id = get_unique_id()


@dataclass
class CreatedUnlearningResources:
    llm_ids: list[int] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)


def _cleanup(created: CreatedUnlearningResources) -> None:
    for run_id in created.run_ids:
        try:
            LlmUnlearningRun.archive(run_id)
        except Exception as error:
            logger.warning(
                "Failed to archive LLM unlearning run with ID %s and exception %s",
                run_id,
                error,
            )
    for llm_id in created.llm_ids:
        try:
            LlmModel.delete_by_id(llm_id)
        except Exception as error:
            logger.warning(
                "Failed to delete LLM with ID %s and exception %s",
                llm_id,
                error,
            )


@pytest.fixture
def created_resources() -> Iterator[CreatedUnlearningResources]:
    created = CreatedUnlearningResources()
    yield created
    _cleanup(created)


def test_unlearn_llm_behavior(created_resources: CreatedUnlearningResources):
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-Qwen3-0.6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    created_resources.llm_ids.append(llm_id)
    LlmRunInfo(target_behaviors=[BiasBehavior()])
    assert llm_id is not None


@pytest.mark.skip(
    reason="SDK-97: backend preprocessing hangs on unlearning runs; re-enable once backend team resolves the issue"
)
def test_unlearn_llm_behavior_full(created_resources: CreatedUnlearningResources):
    if os.getenv("FULL_TEST", "false") != "true":
        pytest.skip("FULL_TEST not enabled")
    llm = LlmModel(
        model_name=f"TEST-UNLEARN-LLM-BEHAVIOR-FULL-Qwen3-0.6B-{unique_id}",
        model_source=HuggingFaceTransformersModel(
            model_name="Qwen/Qwen3-0.6B",
        ),
    )
    llm_id = llm.create()
    created_resources.llm_ids.append(llm_id)
    run_info = LlmRunInfo(target_behaviors=[BiasBehavior()])
    assert llm_id is not None
    run_id = LlmUnlearningRun.launch(llm_id, run_info)
    created_resources.run_ids.append(run_id)
    new_adapter = llm.get_hf_pipeline_for_run(run_id)
    assert isinstance(new_adapter, Pipeline)
