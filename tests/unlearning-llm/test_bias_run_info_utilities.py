from hirundo import BBQBiasType, BiasRunInfo
from hirundo.unlearning_llm import CustomUtility, HuggingFaceDataset


def test_bias_run_info_defaults_to_empty_utility_list() -> None:
    run_info = BiasRunInfo(bias_type=BBQBiasType.ALL).to_run_info()
    assert run_info.target_utilities == []


def test_bias_run_info_forwards_target_utilities() -> None:
    utility = CustomUtility(dataset=HuggingFaceDataset(hugging_face_dataset_name="org/dataset"))
    run_info = BiasRunInfo(bias_type=BBQBiasType.ALL, target_utilities=[utility]).to_run_info()
    assert run_info.target_utilities == [utility]
