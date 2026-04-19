from hirundo import BBQBiasType, BiasRunInfo
from hirundo.unlearning_llm import CustomUtility, HirundoCSVDataset


def test_bias_run_info_defaults_to_empty_utility_list() -> None:
    run_info = BiasRunInfo(bias_type=BBQBiasType.ALL).to_run_info()
    assert run_info.target_utilities == []


def test_bias_run_info_preserves_target_utilities() -> None:
    target_utilities = [
        CustomUtility(
            dataset=HirundoCSVDataset(csv_url="https://example.com/utility.csv")
        )
    ]

    run_info = BiasRunInfo(
        bias_type=BBQBiasType.ALL, target_utilities=target_utilities
    ).to_run_info()

    assert run_info.target_utilities == target_utilities
