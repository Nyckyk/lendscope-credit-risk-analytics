import pandas as pd
import pytest

from src.ingestion import source_contract
from src.modeling.build_expanded_test import read_id_targets
from src.modeling.train_expanded_model import verify_original_cohort


def test_source_schema_accepts_complete_header(monkeypatch):
    columns = sorted(source_contract.REQUIRED_COLUMNS)
    columns += [f"other_{i}" for i in range(source_contract.EXPECTED_COLUMN_COUNT - len(columns))]
    monkeypatch.setattr(source_contract, "read_source_columns", lambda: columns)
    assert source_contract.validate_source_schema() == columns


def test_source_schema_rejects_missing_required_columns(monkeypatch):
    columns = ["other"] * source_contract.EXPECTED_COLUMN_COUNT
    monkeypatch.setattr(source_contract, "read_source_columns", lambda: columns)
    with pytest.raises(ValueError, match="Missing required columns"):
        source_contract.validate_source_schema()


def test_source_schema_rejects_wrong_width(monkeypatch):
    monkeypatch.setattr(source_contract, "read_source_columns", lambda: ["id"])
    with pytest.raises(ValueError, match="Expected 151 columns"):
        source_contract.validate_source_schema()


def test_missing_source_csv_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(source_contract, "SOURCE_PATH", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="Source file not found"):
        source_contract.read_source_columns()


def test_read_id_targets_sorts_and_preserves_outcomes(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"loan_id": [30, 10, 20], "target": [1, 0, 1]}).to_csv(path, index=False)
    series = read_id_targets(path)
    assert series.index.tolist() == [10, 20, 30]
    assert series.tolist() == [0, 1, 1]


@pytest.mark.parametrize("ids,targets,match", [
    ([1, 1], [0, 1], "Duplicate IDs"),
    ([1, None], [0, 1], "Missing IDs"),
    ([1, 2], [0, 2], "Invalid targets"),
])
def test_read_id_targets_rejects_invalid_records(tmp_path, ids, targets, match):
    path = tmp_path / "test.csv"
    pd.DataFrame({"loan_id": ids, "target": targets}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        read_id_targets(path)


def test_original_and_expanded_cohorts_match_independent_of_order(tmp_path, monkeypatch):
    original = pd.DataFrame({"loan_id": [10, 20, 30], "target": [0, 1, 0]})
    original.to_csv(tmp_path / "loan_train.csv", index=False)
    expanded = original.iloc[[2, 0, 1]].copy()
    import src.modeling.train_expanded_model as model_module
    monkeypatch.setattr(model_module, "DATA_DIR", tmp_path)
    verify_original_cohort("train", expanded)


def test_original_and_expanded_cohorts_reject_label_mismatch(tmp_path, monkeypatch):
    pd.DataFrame({"loan_id": [10, 20], "target": [0, 1]}).to_csv(
        tmp_path / "loan_train.csv", index=False
    )
    import src.modeling.train_expanded_model as model_module
    monkeypatch.setattr(model_module, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        verify_original_cohort("train", pd.DataFrame({"loan_id": [10, 20], "target": [1, 1]}))
