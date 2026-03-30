from hirundo import BBQBiasType, BiasRunInfo


def test_bias_run_info_defaults_to_empty_utility_list() -> None:
    run_info = BiasRunInfo(bias_type=BBQBiasType.ALL).to_run_info()
    assert run_info.target_utilities == []
