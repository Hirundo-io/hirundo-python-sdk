import secrets

import hirundo.llm_behavior_eval as llm_behavior_eval_module
import pytest
from hirundo._llm_sources import HuggingFaceTransformersModelOutput
from hirundo.llm_behavior_eval import EvalRunInfo, LlmBehaviorEval, ModelOrRun


def test_validate_model_access_forwards_huggingface_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    captured_call: dict[str, str | None] = {}

    monkeypatch.setattr(
        llm_behavior_eval_module.LlmModel,
        "get_by_id",
        lambda model_id: llm_model_output,
    )

    def fake_validate_huggingface_model_access(
        model_name: str, token: str | None, model_role: str
    ) -> None:
        captured_call.update(
            {
                "model_name": model_name,
                "token": token,
                "model_role": model_role,
            }
        )

    monkeypatch.setattr(
        llm_behavior_eval_module,
        "validate_huggingface_model_access",
        fake_validate_huggingface_model_access,
    )

    LlmBehaviorEval._validate_model_access(ModelOrRun.MODEL, run_info)

    assert captured_call == {
        "model_name": "org/private-model",
        "token": user_access_secret,
        "model_role": "LLM",
    }
