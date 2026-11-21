
import json

import joblib
import numpy as np
import pandas as pd

from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from src.modeling.feature_ablation import REDUCED_FEATURES
from src.modeling.train_expanded_model import DATA_DIR, REPORT_DIR
from src.modeling.train_baseline import evaluate


MODEL_PATH = (
    DATA_DIR / "hist_gradient_boosting_reduced.joblib"
)

VALIDATION_PATH = (
    DATA_DIR / "loan_validation_expanded.csv"
)

CALIBRATOR_PATH = (
    DATA_DIR / "risk_calibrator.joblib"
)

REPORT_PATH = (
    REPORT_DIR / "calibration_validation.json"
)


def print_results(name, y, probabilities):
    metrics = evaluate(y, probabilities)

    print(f"\n{name}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )
    print(f"Brier score:       {metrics['brier_score']:.5f}")
    print(f"Mean predicted:    {probabilities.mean():.2%}")
    print(f"Mean observed:     {y.mean():.2%}")

    return metrics


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reduced Gradient Boosting...", flush=True)

    model = joblib.load(MODEL_PATH)

    print("Loading 2014 validation data...", flush=True)

    df = pd.read_csv(
        VALIDATION_PATH,
        usecols=["issue_date", "target"] + REDUCED_FEATURES,
        parse_dates=["issue_date"],
    )

    if df["issue_date"].isna().any():
        raise ValueError("Missing issue dates")

    if not df["issue_date"].dt.year.eq(2014).all():
        raise ValueError("Expected only 2014 loans")

    if not df["target"].isin([0, 1]).all():
        raise ValueError("Invalid target values")

    # First half is for fitting the calibrator.
    calibration_mask = (
        df["issue_date"] < pd.Timestamp("2014-07-01")
    ).to_numpy()

    # Second half is for evaluating the adjustment.
    evaluation_mask = ~calibration_mask

    y = df["target"].astype(int).to_numpy()

    if (
        len(np.unique(y[calibration_mask])) != 2
        or len(np.unique(y[evaluation_mask])) != 2
    ):
        raise ValueError(
            "Both periods require positive and negative cases"
        )

    print(
        f"Calibration loans: {calibration_mask.sum():,}"
    )
    print(
        f"Evaluation loans: {evaluation_mask.sum():,}"
    )

    print("Generating original predictions...", flush=True)

    raw_probabilities = model.predict_proba(
        df[REDUCED_FEATURES]
    )[:, 1]

    # Convert probability estimates to log-odds.
    # Clipping prevents infinite log-odds for scores
    # exactly equal to zero or one.
    safe_probabilities = np.clip(
        raw_probabilities,
        1e-6,
        1 - 1e-6,
    )

    scores = logit(safe_probabilities).reshape(-1, 1)

    print("Fitting Platt calibrator...", flush=True)

    calibrator = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    calibrator.fit(
        scores[calibration_mask],
        y[calibration_mask],
    )

    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError(
            "Calibration reversed the risk ranking"
        )

    # Compare on July-December only.
    y_evaluation = y[evaluation_mask]

    raw_evaluation = raw_probabilities[
        evaluation_mask
    ]

    calibrated_evaluation = (
        calibrator.predict_proba(
            scores[evaluation_mask]
        )[:, 1]
    )

    print("\n=== CALIBRATION EVALUATION ===")

    before = print_results(
        "Original model",
        y_evaluation,
        raw_evaluation,
    )

    after = print_results(
        "Calibrated model",
        y_evaluation,
        calibrated_evaluation,
    )

    print("\n=== CALIBRATION PARAMETERS ===")
    print(
        f"Slope: {calibrator.coef_[0, 0]:.5f}"
    )
    print(
        f"Intercept: {calibrator.intercept_[0]:.5f}"
    )

    report = {
        "model": "12-feature Histogram Gradient Boosting",
        "calibration_method": "Platt scaling on log-odds",
        "calibration_period": "January-June 2014",
        "evaluation_period": "July-December 2014",
        "calibration_loans": int(calibration_mask.sum()),
        "evaluation_loans": int(evaluation_mask.sum()),
        "before": before,
        "after": after,
        "calibration_slope": float(
            calibrator.coef_[0, 0]
        ),
        "calibration_intercept": float(
            calibrator.intercept_[0]
        ),
        "independent_model_selection_test": False,
        "test_set_evaluated": False,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    joblib.dump(calibrator, CALIBRATOR_PATH)

    print(f"\nCalibrator saved: {CALIBRATOR_PATH}")
    print(f"Report saved: {REPORT_PATH}")

    print("\n2015 test data remains untouched.")


if __name__ == "__main__":
    main()
