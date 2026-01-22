import importlib.util
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hirundo import HirundoError
from hirundo._http import requests
from hirundo._timeouts import DOWNLOAD_READ_TIMEOUT
from hirundo.logger import get_logger

if TYPE_CHECKING:
    from torch import device as torch_device
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pipelines.base import Pipeline

    from hirundo.unlearning_llm import LlmModel, LlmModelOut

logger = get_logger(__name__)


ZIP_FILE_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB
REQUIRED_PACKAGES_FOR_PIPELINE = ["peft", "transformers", "accelerate"]


def get_hf_pipeline_for_run_given_model(
    llm: "LlmModel | LlmModelOut",
    run_id: str,
    config: "PretrainedConfig | None" = None,
    device: "str | int | torch_device | None" = None,
    device_map: str | dict[str, int | str] | None = None,
    trust_remote_code: bool = False,
    token: str | None = None,
) -> "Pipeline":
    for package in REQUIRED_PACKAGES_FOR_PIPELINE:
        if importlib.util.find_spec(package) is None:
            raise HirundoError(
                f'{package} is not installed. Please install transformers extra with pip install "hirundo[transformers]"'
            )
    from peft import PeftModel
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.pipelines import pipeline

    from hirundo.unlearning_llm import (
        HuggingFaceTransformersModel,
        HuggingFaceTransformersModelOutput,
        LlmUnlearningRun,
    )

    run_results = LlmUnlearningRun.check_run_by_id(run_id)
    if run_results is None:
        raise HirundoError("No run results found")
    result_payload = (
        run_results.get("result", run_results)
        if isinstance(run_results, dict)
        else run_results
    )
    if isinstance(result_payload, dict):
        result_url = result_payload.get("result")
    else:
        result_url = result_payload
    if not isinstance(result_url, str):
        raise HirundoError("Run results did not include a download URL")
    # Stream the zip file download

    zip_file_path = tempfile.NamedTemporaryFile(delete=False).name
    with requests.get(
        result_url,
        timeout=DOWNLOAD_READ_TIMEOUT,
        stream=True,
    ) as r:
        r.raise_for_status()
        with open(zip_file_path, "wb") as zip_file:
            for chunk in r.iter_content(chunk_size=ZIP_FILE_CHUNK_SIZE):
                zip_file.write(chunk)
        logger.info(
            "Successfully downloaded the result zip file for run ID %s to %s",
            run_id,
            zip_file_path,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            zip_file.extractall(temp_dir_path)
        # Attempt to load the tokenizer normally
        base_model_name = (
            llm.model_source.model_name
            if isinstance(
                llm.model_source,
                HuggingFaceTransformersModel | HuggingFaceTransformersModelOutput,
            )
            else llm.model_source.local_path
        )
        token = (
            llm.model_source.token
            if isinstance(
                llm.model_source,
                HuggingFaceTransformersModel,
            )
            else token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(
            base_model_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        config_dict = config.to_dict() if hasattr(config, "to_dict") else config
        is_multimodal = (
            config_dict.get("model_type") in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        )
        if is_multimodal:
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        model = cast(
            "PreTrainedModel",
            PeftModel.from_pretrained(
                base_model, str(temp_dir_path / "unlearned_model_folder")
            ),
        )

        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            device_map=device_map,
        )
