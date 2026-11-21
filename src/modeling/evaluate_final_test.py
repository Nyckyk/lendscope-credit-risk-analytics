
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from scipy.special import logit

from src.modeling.train_baseline import (
    DATA_DIR,
    PROJECT_ROOT,
    FEATURES as BASE_FEATURES,
    evaluate,
)

from src.modeling.feature_ablation import REDUCED_FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

EXPECTED_ROWS = 283_026
EXPECTED_CHARGED_OFF = 42_132

ORIGINAL_TEST = DATA_DIR / "loan_test.csv"
EXPANDED_TEST = DATA_DIR / "loan_test_expanded.csv"

ORIGINAL_MODEL = (
    DATA_DIR / "hist_gradient_boosting.joblib"
)

REDUCED_MODEL = (
    DATA_DIR / "hist_gradient_boosting_reduced.joblib"
)

CALIBRATOR = DATA_DIR / "risk_calibrator.joblib"

REPORT_PATH = REPORT_DIR / "final_test_evaluation.json"
DECILES_PATH = REPORT_DIR / "final_test_deciles.csv"


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_verify_test_data():

    print("Loading 2015 test datasets...", flush=True)

    expanded = pd.read_csv(
        EXPANDED_TEST,
        usecols=[
            "loan_id",
            "issue_date",
            "target",
        ] + REDUCED_FEATURES,
        parse_dates=["issue_date"],
    )

    original = pd.read_csv(
        ORIGINAL_TEST,
        usecols=[
            "loan_id",
            "target",
        ] + BASE_FEATURES,
    )

    for name, df in [
        ("Original", original),
        ("Expanded", expanded),
    ]:
        if len(df) != EXPECTED_ROWS:
            raise ValueError(
                f"{name}: unexpected test row count"
            )

        if df["loan_id"].isna().any():
            raise ValueError(
                f"{name}: missing loan IDs"
            )

        if not df["loan_id"].is_unique:
            raise ValueError(
                f"{name}: duplicate loan IDs"
            )

        if not df["target"].isin([0, 1]).all():
            raise ValueError(
                f"{name}: invalid target values"
            )

        if int(df["target"].sum()) != EXPECTED_CHARGED_OFF:
            raise ValueError(
                f"{name}: incorrect charged-off count"
            )

    if expanded["issue_date"].isna().any():
        raise ValueError("Missing issue dates")

    if not expanded["issue_date"].dt.year.eq(2015).all():
        raise ValueError("Test data contains non-2015 loans")

    # Align original six-feature data to the expanded
    # test dataset using loan IDs, not row position.
    original = (
        original
        .set_index("loan_id")
        .loc[expanded["loan_id"]]
        .reset_index(drop=True)
    )

    if not np.array_equal(
        original["target"].to_numpy(),
        expanded["target"].to_numpy(),
    ):
        raise ValueError(
            "Original and expanded targets do not match"
        )

    print("Test identities and targets verified.")
    print(f"Test loans: {len(expanded):,}")
    print(f"Charged off: {EXPECTED_CHARGED_OFF:,}")

    return original, expanded


# ============================================================
# PREDICTION AND METRICS
# ============================================================

def predict(model_path, features):

    model = joblib.load(model_path)

    probabilities = model.predict_proba(features)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Invalid predictions from {model_path.name}"
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"Probabilities outside [0, 1] from "
            f"{model_path.name}"
        )

    return probabilities


def calculate_deciles(y, probabilities, model_name):

    results = pd.DataFrame({
        "actual": y,
        "predicted": probabilities,
    })

    # Decile 1: lowest predicted risk.
    # Decile 10: highest predicted risk.
    results["decile"] = (
        pd.qcut(
            results["predicted"].rank(method="first"),
            q=10,
            labels=False,
        ) + 1
    )

    summary = (
        results.groupby("decile", as_index=False)
        .agg(
            loans=("actual", "size"),
            charged_off=("actual", "sum"),
            mean_prediction=("predicted", "mean"),
            observed_rate=("actual", "mean"),
        )
    )

    summary["predicted_pct"] = (
        summary["mean_prediction"] * 100
    )

    summary["observed_pct"] = (
        summary["observed_rate"] * 100
    )

    summary["model"] = model_name

    return summary


def score_model(name, y, probabilities):

    metrics = evaluate(y, probabilities)
    deciles = calculate_deciles(y, probabilities, name)

    lowest = float(deciles.iloc[0]["observed_rate"])
    highest = float(deciles.iloc[-1]["observed_rate"])

    risk_ratio = (
        highest / lowest
        if lowest > 0
        else None
    )

    top_decile_capture = (
        float(deciles.iloc[-1]["charged_off"])
        / float(y.sum())
        * 100
    )

    metrics.update({
        "mean_predicted_pct": float(
            probabilities.mean() * 100
        ),
        "observed_pct": float(y.mean() * 100),
        "prediction_bias_pp": float(
            (probabilities.mean() - y.mean()) * 100
        ),
        "top_bottom_risk_ratio": risk_ratio,
        "top_decile_capture_pct": top_decile_capture,
    })

    return metrics, deciles


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 58)
    print("LENDSCOPE: FINAL 2015 HOLDOUT EVALUATION")
    print("=" * 58)

    original, expanded = load_and_verify_test_data()

    y_test = expanded["target"].astype(int).to_numpy()

    # --------------------------------------------------------
    # Model 1: Original six-feature Gradient Boosting
    # --------------------------------------------------------

    print(
        "\nEvaluating original six-feature model...",
        flush=True,
    )

    original_probs = predict(
        ORIGINAL_MODEL,
        original[BASE_FEATURES],
    )

    # --------------------------------------------------------
    # Model 2: Selected 12-feature Gradient Boosting
    # --------------------------------------------------------

    print(
        "Evaluating selected 12-feature model...",
        flush=True,
    )

    reduced_probs = predict(
        REDUCED_MODEL,
        expanded[REDUCED_FEATURES],
    )

    # --------------------------------------------------------
    # Model 3: Same model with frozen Platt calibration
    # --------------------------------------------------------

    print(
        "Applying frozen Platt calibrator...",
        flush=True,
    )

    calibrator = joblib.load(CALIBRATOR)

    if calibrator.coef_[0, 0] <= 0:
        raise ValueError(
            "Calibrator does not preserve risk ranking"
        )

    # Reproduce the exact transformation used when
    # fitting the calibrator on January-June 2014.
    safe_probs = np.clip(
        reduced_probs,
        1e-6,
        1 - 1e-6,
    )

    log_odds = logit(safe_probs).reshape(-1, 1)

    calibrated_probs = (
        calibrator.predict_proba(log_odds)[:, 1]
    )

    predictions = {
        "Original 6-feature": original_probs,
        "Reduced 12-feature": reduced_probs,
        "Reduced + Platt": calibrated_probs,
    }

    # --------------------------------------------------------
    # Compare models on identical 2015 loans
    # --------------------------------------------------------

    print("\n=== FINAL TEST RESULTS ===")

    reports = {}
    all_deciles = []

    for name, probabilities in predictions.items():

        metrics, deciles = score_model(
            name,
            y_test,
            probabilities,
        )

        reports[name] = metrics
        all_deciles.append(deciles)

        print(f"\n{name}")
        print(
            f"ROC-AUC:              "
            f"{metrics['roc_auc']:.4f}"
        )
        print(
            f"Average Precision:    "
            f"{metrics['average_precision']:.4f}"
        )
        print(
            f"Brier score:          "
            f"{metrics['brier_score']:.5f}"
        )
        print(
            f"Mean predicted:       "
            f"{metrics['mean_predicted_pct']:.2f}%"
        )
        print(
            f"Observed charge-off:  "
            f"{metrics['observed_pct']:.2f}%"
        )
        print(
            f"Prediction bias:      "
            f"{metrics['prediction_bias_pp']:+.2f} pp"
        )
        print(
            f"Top/bottom risk ratio: "
            f"{metrics['top_bottom_risk_ratio']:.2f}x"
        )
        print(
            f"Top-decile capture:   "
            f"{metrics['top_decile_capture_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Display calibrated risk deciles
    # --------------------------------------------------------

    decile_results = pd.concat(
        all_deciles,
        ignore_index=True,
    )

    calibrated_deciles = decile_results[
        decile_results["model"] == "Reduced + Platt"
    ]

    print("\n=== CALIBRATED MODEL TEST RISK DECILES ===")

    print(
        calibrated_deciles[
            [
                "decile",
                "loans",
                "charged_off",
                "predicted_pct",
                "observed_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    report = {
        "evaluated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "test_year": 2015,
        "test_loans": len(y_test),
        "charged_off_loans": int(y_test.sum()),
        "outcome_definition": (
            "Charged Off versus Fully Paid, "
            "resolved 36-month LendingClub loans"
        ),
        "training_years": "2011-2013",
        "model_selection_year": 2014,
        "calibration_period": "January-June 2014",
        "calibration_check": "July-December 2014",
        "models": reports,
        "notes": [
            "Models and calibrator were not refitted on test data.",
            "Test cohort excludes loans with unresolved outcomes.",
            "Retrospective historical-outcome evaluation.",
            "Not a fully point-in-time lending backtest.",
            "Not validated for production underwriting decisions.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    decile_results.to_csv(
        DECILES_PATH,
        index=False,
    )

    print("\n" + "=" * 58)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 58)

    print(f"Metrics report: {REPORT_PATH}")
    print(f"Decile report: {DECILES_PATH}")

    print(
        "\nDo not change model-selection decisions "
        "based on these final test results."
    )


if __name__ == "__main__":
    main()
