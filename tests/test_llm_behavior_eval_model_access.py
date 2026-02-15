import secrets
from unittest.mock import patch

from hirundo._llm_sources import HuggingFaceTransformersModelOutput
from hirundo.llm_behavior_eval import EvalRunInfo, LlmBehaviorEval, ModelOrRun


def test_validate_model_access_forwards_huggingface_token() -> None:
    run_info = EvalRunInfo(model_id=123)
    user_access_secret = secrets.token_hex(8)
    llm_model_output = type(
        "LlmModelStub",
        (),
        {
            "model_source": HuggingFaceTransformersModelOutput(
                model_name="org/private-model",
                token=user_access_secret,
            )
        },
    )()

    with (
        patch(
            "hirundo.llm_behavior_eval.LlmModel.get_by_id",
            return_value=llm_model_output,
        ),
        patch(
            "hirundo.llm_behavior_eval.validate_huggingface_model_access"
        ) as mock_validate_huggingface_model_access,
    ):
        LlmBehaviorEval._validate_model_access(ModelOrRun.MODEL, run_info)

    mock_validate_huggingface_model_access.assert_called_once_with(
        model_name="org/private-model",
        token=user_access_secret,
        model_role="LLM",
    )
