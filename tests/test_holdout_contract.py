"""Holdout data contracts only: small synthetic CSVs, never real 2015 data."""

import pandas as pd
import pytest

import src.modeling.evaluate_final_test as holdout


@pytest.fixture
def synthetic_holdout(tmp_path, monkeypatch):
    # The exact same columns used by the holdout loader, but only three records.
    ids = [10, 20, 30]
    targets = {10: 0, 20: 1, 30: 0}
    original = pd.DataFrame({"loan_id": [30, 10, 20], "target": [targets[i] for i in [30, 10, 20]]})
    expanded = pd.DataFrame({"loan_id": ids, "target": [targets[i] for i in ids],
                             "issue_date": ["2015-01-01", "2015-06-01", "2015-11-01"]})
    for col in holdout.BASE_FEATURES:
        original[col] = "RENT" if col == "home_ownership" else ("debt_consolidation" if col == "loan_purpose" else 10.0)
    original["loan_amount"] = original["loan_id"]  # Distinct synthetic values expose misalignment.
    for col in holdout.REDUCED_FEATURES:
        expanded[col] = "RENT" if col == "home_ownership" else (
            "debt_consolidation" if col == "loan_purpose" else ("1 year" if col == "employment_length" else 10.0)
        )
    original_path = tmp_path / "original.csv"
    expanded_path = tmp_path / "expanded.csv"
    original.to_csv(original_path, index=False)
    expanded.to_csv(expanded_path, index=False)
    monkeypatch.setattr(holdout, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(holdout, "EXPECTED_CHARGED_OFF", 1)
    monkeypatch.setattr(holdout, "ORIGINAL_TEST", original_path)
    monkeypatch.setattr(holdout, "EXPANDED_TEST", expanded_path)
    return original_path, expanded_path


def test_joins_original_features_by_loan_id_not_row(synthetic_holdout):
    original, expanded = holdout.load_and_verify_test_data()
    assert expanded["loan_id"].tolist() == [10, 20, 30]
    # Loader intentionally drops original IDs after alignment; match via unique amounts.
    assert original["loan_amount"].tolist() == [10, 20, 30]
    assert original["target"].tolist() == expanded["target"].tolist()


def test_holdout_rejects_wrong_issue_year(synthetic_holdout):
    _, path = synthetic_holdout
    df = pd.read_csv(path)
    df.loc[0, "issue_date"] = "2014-12-01"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-2015"):
        holdout.load_and_verify_test_data()


def test_holdout_rejects_duplicate_loan_ids(synthetic_holdout):
    _, path = synthetic_holdout
    df = pd.read_csv(path)
    df.loc[2, "loan_id"] = df.loc[1, "loan_id"]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate loan IDs"):
        holdout.load_and_verify_test_data()


def test_holdout_rejects_wrong_target_count(synthetic_holdout):
    _, path = synthetic_holdout
    df = pd.read_csv(path)
    df["target"] = 0
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="incorrect charged-off count"):
        holdout.load_and_verify_test_data()


def test_holdout_rejects_mismatched_labels(synthetic_holdout):
    original_path, _ = synthetic_holdout
    df = pd.read_csv(original_path)
    # Preserve 1 charged-off loan, but move it to a different ID.
    df["target"] = [1, 0, 0]
    df.to_csv(original_path, index=False)
    with pytest.raises(ValueError, match="targets do not match"):
        holdout.load_and_verify_test_data()
