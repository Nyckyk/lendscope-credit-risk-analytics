import pandas as pd
import pytest

from src.modeling.build_expanded_dataset import NEW_NUMERIC, transform_chunk


def test_valid_transformation_and_missing_categories(raw_feature_chunk):
    output = transform_chunk(raw_feature_chunk)

    assert output["credit_history_months"].tolist() == [48, 1]
    assert "earliest_cr_line" not in output.columns
    assert output["employment_length"].tolist() == ["10+ years", "Unknown"]
    assert output["verification_status"].tolist() == ["Verified", "Unknown"]
    assert output["inq_last_6mths"].iloc[0] == 2
    assert pd.isna(output["inq_last_6mths"].iloc[1])
    assert pd.isna(output["revol_util"].iloc[1])
    assert output["target"].tolist() == [0, 1]


def test_transformation_does_not_modify_input(raw_feature_chunk):
    original = raw_feature_chunk.copy(deep=True)
    transform_chunk(raw_feature_chunk)
    pd.testing.assert_frame_equal(raw_feature_chunk, original)


@pytest.mark.parametrize("field", NEW_NUMERIC)
def test_rejects_unexpected_non_numeric_values(raw_feature_chunk, field):
    raw_feature_chunk.loc[0, field] = "not-a-number"
    with pytest.raises(ValueError, match=f"Invalid values in {field}"):
        transform_chunk(raw_feature_chunk)


@pytest.mark.parametrize("field,value,message", [
    ("earliest_cr_line", "not-a-date", "Invalid earliest credit line"),
    ("issue_date", "bad-date", "Invalid loan issue dates"),
    ("earliest_cr_line", "Jan-2015", "Credit history begins after"),
    ("target", 2, "Unexpected target values"),
])
def test_rejects_inconsistent_records(raw_feature_chunk, field, value, message):
    raw_feature_chunk.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        transform_chunk(raw_feature_chunk)


def test_100_percent_utilisation_is_not_arbitrarily_capped(raw_feature_chunk):
    raw_feature_chunk.loc[0, "revol_util"] = "125.5"
    assert transform_chunk(raw_feature_chunk).loc[0, "revol_util"] == 125.5
