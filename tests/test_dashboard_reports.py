"""Unit-test the actual app.py report loader functions without starting Streamlit.

We compile only the two pure loader function AST nodes. This prevents importing
Streamlit, creating a database connection, or executing dashboard UI code.
It is intentionally NOT a six-page browser or Streamlit rendering test.
"""

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.modeling.feature_ablation import REDUCED_FEATURES


@pytest.fixture
def dashboard_loaders():
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    wanted = {"load_risk_reports", "load_reduced_importance"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in functions} != wanted:
        pytest.fail("Could not find both pure risk-report loaders in dashboard/app.py")

    environment = {
        "math": math,
        "json": json,
        "hashlib": hashlib,
        "pd": pd,
        "MODEL_NAMES": ("Original 6-feature", "Reduced 12-feature", "Reduced + Platt"),
    }
    code = compile(ast.Module(body=functions, type_ignores=[]), str(app_path), "exec")
    exec(code, environment)
    return SimpleNamespace(**{name: environment[name] for name in wanted}, env=environment)


@pytest.fixture
def risk_reports(tmp_path, dashboard_loaders):
    folder = tmp_path / "reports" / "modeling"
    folder.mkdir(parents=True)
    dashboard_loaders.env["MODELING_REPORT_DIR"] = folder

    names = dashboard_loaders.env["MODEL_NAMES"]
    scores = [(0.6419, 0.2230, 0.12302), (0.6565, 0.2364, 0.12238), (0.6565, 0.2364, 0.12222)]
    loans = [28303, 28303, 28302, 28303, 28302, 28303, 28302, 28303, 28302, 28303]
    charged = [1159, 2129, 2715, 3275, 3785, 4264, 4960, 5413, 6442, 7990]
    assert sum(loans) == 283026 and sum(charged) == 42132
    predictions = [4.16, 6.62, 8.27, 9.78, 11.32, 12.96, 14.83, 17.03, 19.91, 25.57]
    pred_mean = sum(p * n for p, n in zip(predictions, loans)) / sum(loans)
    observed_mean = sum(charged) / sum(loans) * 100

    deciles = []
    for name in names:
        for i, (n, c, p) in enumerate(zip(loans, charged, predictions), start=1):
            deciles.append({"model": name, "decile": i, "loans": n, "charged_off": c,
                            "predicted_pct": p, "observed_pct": 100 * c / n})
    decile_path = folder / "final_test_deciles.csv"
    pd.DataFrame(deciles).to_csv(decile_path, index=False, float_format="%.14f")

    models = {}
    for name, (auc, ap, brier) in zip(names, scores):
        models[name] = {"roc_auc": auc, "average_precision": ap, "brier_score": brier,
                        "mean_predicted_pct": pred_mean, "observed_pct": observed_mean,
                        "prediction_bias_pp": pred_mean - observed_mean}
    report = {"evaluated_at_utc": "2026-11-02T02:00:00+00:00", "test_year": 2015,
              "test_loans": 283026, "charged_off_loans": 42132, "models": models}
    report_path = folder / "final_test_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return SimpleNamespace(folder=folder, report=report, report_path=report_path, decile_path=decile_path,
                           loaders=dashboard_loaders)


def test_valid_holdout_report_loads(risk_reports):
    report, deciles = risk_reports.loaders.load_risk_reports()
    assert report["test_loans"] == 283026
    assert len(deciles) == 30
    assert set(deciles["model"]) == set(risk_reports.loaders.env["MODEL_NAMES"])


def test_missing_holdout_report_fails_cleanly(risk_reports):
    risk_reports.report_path.unlink()
    with pytest.raises(FileNotFoundError):
        risk_reports.loaders.load_risk_reports()


def test_holdout_rejects_wrong_cohort_size(risk_reports):
    risk_reports.report["test_loans"] += 1
    risk_reports.report_path.write_text(json.dumps(risk_reports.report), encoding="utf-8")
    with pytest.raises(ValueError, match="verified 2015 cohort"):
        risk_reports.loaders.load_risk_reports()


def test_holdout_rejects_non_finite_metric(risk_reports):
    risk_reports.report["models"]["Reduced + Platt"]["roc_auc"] = float("nan")
    risk_reports.report_path.write_text(json.dumps(risk_reports.report), encoding="utf-8")
    with pytest.raises(ValueError, match="finite JSON number"):
        risk_reports.loaders.load_risk_reports()


def test_holdout_rejects_non_reconciling_decile_counts(risk_reports):
    frame = pd.read_csv(risk_reports.decile_path)
    frame.loc[0, "charged_off"] += 1
    frame.to_csv(risk_reports.decile_path, index=False)
    with pytest.raises(ValueError, match="Decile counts or percentages|Observed decile rates"):
        risk_reports.loaders.load_risk_reports()


def test_holdout_rejects_inconsistent_mean_prediction(risk_reports):
    risk_reports.report["models"]["Reduced + Platt"]["mean_predicted_pct"] += 1
    risk_reports.report_path.write_text(json.dumps(risk_reports.report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reconcile"):
        risk_reports.loaders.load_risk_reports()


@pytest.fixture
def importance_reports(risk_reports):
    folder = risk_reports.folder
    features = list(REDUCED_FEATURES)
    frame = pd.DataFrame({"feature": features,
                          "mean_auc_drop": np.linspace(0.001, 0.06, len(features)),
                          "std_auc_drop": [0.001] * len(features)})
    csv_path = folder / "reduced_feature_importance.csv"
    frame.to_csv(csv_path, index=False)
    meta = {"report_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "reference_auc": 0.6485, "model": "Reduced 12-feature", "evaluation_year": 2014,
            "scoring": "roc_auc", "sample_size": 20000, "n_repeats": 3,
            "features": features}
    (folder / "reduced_feature_importance_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "feature_ablation_validation.json").write_text(json.dumps({"reduced_features": features}), encoding="utf-8")
    return SimpleNamespace(folder=folder, csv_path=csv_path, meta=meta, loaders=risk_reports.loaders)


def test_valid_permutation_importance_loads(importance_reports):
    frame, meta = importance_reports.loaders.load_reduced_importance()
    assert len(frame) == 12
    assert frame["mean_auc_drop"].is_monotonic_increasing
    assert meta["reference_auc"] == 0.6485


def test_importance_file_tampering_is_detected(importance_reports):
    with importance_reports.csv_path.open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        importance_reports.loaders.load_reduced_importance()


def test_importance_rejects_negative_shuffle_deviation(importance_reports):
    frame = pd.read_csv(importance_reports.csv_path)
    frame.loc[0, "std_auc_drop"] = -0.01
    frame.to_csv(importance_reports.csv_path, index=False)
    importance_reports.meta["report_sha256"] = hashlib.sha256(importance_reports.csv_path.read_bytes()).hexdigest()
    (importance_reports.folder / "reduced_feature_importance_metadata.json").write_text(
        json.dumps(importance_reports.meta), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid permutation importance"):
        importance_reports.loaders.load_reduced_importance()
