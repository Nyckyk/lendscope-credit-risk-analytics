
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.modeling.feature_ablation import REDUCED_FEATURES
from src.modeling.train_expanded_model import (
    DATA_DIR,
    REPORT_DIR,
    load_expanded,
)


MODEL_PATH = (
    DATA_DIR / "hist_gradient_boosting_reduced.joblib"
)


def examine_group(name, mask, actual, probabilities):
    """Evaluate model performance on one loan subgroup."""

    mask = mask.fillna(False).astype(bool)

    y = actual.loc[mask]
    p = probabilities[mask.to_numpy()]

    if len(y) == 0:
        print(f"\n{name}: no loans")
        return None

    result = {
        "group": name,
        "loans": int(len(y)),
        "charged_off": int(y.sum()),
        "observed_rate": float(y.mean()),
        "predicted_rate": float(p.mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": None,
        "average_precision": None,
    }

    # Small groups can produce unstable discrimination
    # estimates. Report their counts and rates, but skip
    # ROC-AUC and AP if fewer than 100 loans are present.
    if len(y) >= 100 and y.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(
            average_precision_score(y, p)
        )

    print(f"\n{name}")
    print(f"Loans: {result['loans']:,}")
    print(f"Charged off: {result['charged_off']:,}")
    print(f"Observed rate: {result['observed_rate']:.2%}")
    print(f"Mean predicted: {result['predicted_rate']:.2%}")
    print(f"Brier score: {result['brier_score']:.4f}")

    if result["roc_auc"] is not None:
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(
            f"Average Precision: "
            f"{result['average_precision']:.4f}"
        )
    else:
        print("ROC-AUC / AP: insufficient observations")

    return result


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training and validation data...", flush=True)

    train = load_expanded("train")
    validation = load_expanded("validation")

    print("Loading reduced Gradient Boosting model...", flush=True)

    model = joblib.load(MODEL_PATH)

    # Derive income thresholds using training data only.
    income_p99 = float(
        train["annual_income"].quantile(0.99)
    )

    income_p995 = float(
        train["annual_income"].quantile(0.995)
    )

    util_p99 = float(
        train["revol_util"].quantile(0.99)
    )

    print("\n=== TRAINING DISTRIBUTION ===")
    print(f"Income 99th percentile: ${income_p99:,.2f}")
    print(f"Income 99.5th percentile: ${income_p995:,.2f}")
    print(f"Utilisation 99th percentile: {util_p99:.2f}%")

    print("\n=== OUTLIER PREVALENCE ===")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
    ]:
        income_flag = df["annual_income"] > income_p99
        util_flag = df["revol_util"] > 100

        print(f"\n{name}")
        print(f"Loans: {len(df):,}")
        print(
            f"Income above training P99: "
            f"{int(income_flag.sum()):,}"
        )
        print(
            f"Utilisation above 100%: "
            f"{int(util_flag.sum()):,}"
        )
        print(
            f"Missing utilisation: "
            f"{int(df['revol_util'].isna().sum()):,}"
        )

    print("\nGenerating validation predictions...", flush=True)

    X_val = validation[REDUCED_FEATURES]
    y_val = validation["target"].astype(int)

    probabilities = model.predict_proba(X_val)[:, 1]

    income_flag = (
        validation["annual_income"] > income_p99
    ).fillna(False)

    util_flag = (
        validation["revol_util"] > 100
    ).fillna(False)

    all_loans = pd.Series(
        True,
        index=validation.index,
    )

    # Groups overlap: a loan may have both high income
    # and high credit utilisation.
    groups = [
        ("All validation loans", all_loans),
        ("Income above training P99", income_flag),
        ("Revolving utilisation above 100%", util_flag),
        ("Neither extreme flagged", ~(income_flag | util_flag)),
    ]

    print("\n=== MODEL ROBUSTNESS RESULTS ===")

    results = []

    for name, mask in groups:
        result = examine_group(
            name,
            mask,
            y_val,
            probabilities,
        )

        if result is not None:
            results.append(result)

    report = {
        "model": "12-feature Gradient Boosting",
        "training_years": "2011-2013",
        "validation_year": 2014,
        "thresholds_from_training": {
            "income_p99": income_p99,
            "income_p995": income_p995,
            "revol_util_p99": util_p99,
        },
        "groups": results,
        "test_set_evaluated": False,
    }

    output_path = (
        REPORT_DIR / "outlier_robustness_validation.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nReport saved: {output_path}")
    print("2015 test data was not accessed.")


if __name__ == "__main__":
    main()
