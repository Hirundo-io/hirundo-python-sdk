def validate_column_options(
    *,
    feature_cols: list[str] | None,
    extra_non_feature_cols: list[str] | None,
    modality: object,
    allowed_modalities: frozenset[object],
    unsupported_message: str,
) -> None:
    """Validate mutually exclusive Dataset QA column mode options.

    Args:
        feature_cols: Feature column names selected for model input, if provided.
        extra_non_feature_cols: Non-feature column names to preserve in outputs, if provided.
        modality: Dataset or child modality being validated.
        allowed_modalities: Modalities that support column mode options.
        unsupported_message: Error message to use when column options are provided
            for an unsupported modality.

    Returns:
        None. Raises ValueError when both column modes are provided or the modality
        does not support column mode options.
    """
    has_feature_cols = feature_cols is not None
    has_extra_non_feature_cols = extra_non_feature_cols is not None
    if not has_feature_cols and not has_extra_non_feature_cols:
        return
    if has_feature_cols and has_extra_non_feature_cols:
        raise ValueError(
            "Only one of `feature_cols` or `extra_non_feature_cols` can be provided"
        )
    if modality not in allowed_modalities:
        raise ValueError(unsupported_message)
