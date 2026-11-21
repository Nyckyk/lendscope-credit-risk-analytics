
from pathlib import Path

import joblib
import pandas as pd

from src.modeling.train_baseline import (
    DATA_DIR,
    FEATURES,
    PROJECT_ROOT,
    evaluate,
)

REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

MODELS = {
    "Logistic Regression": "logistic_baseline.joblib",
    "Gradient Boosting": "hist_gradient_boosting.joblib",
}


def analyse_model(name, model, X, y):
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "actual": y.to_numpy(),
        "predicted": probabilities,
    })

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
            predicted_rate=("predicted", "mean"),
            observed_rate=("actual", "mean"),
        )
    )

    summary["predicted_pct"] = (
        100 * summary["predicted_rate"]
    )

    summary["observed_pct"] = (
        100 * summary["observed_rate"]
    )

    metrics = evaluate(y, probabilities)

    lowest = summary.iloc[0]["observed_rate"]
    highest = summary.iloc[-1]["observed_rate"]

    risk_ratio = (
        highest / lowest if lowest > 0 else float("nan")
    )

    top_decile_capture = (
        summary.iloc[-1]["charged_off"]
        / y.sum() * 100
    )

    print(f"\n=== {name.upper()} ===")

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(
        f"Average precision: "
        f"{metrics['average_precision']:.4f}"
    )
    print(f"Brier score: {metrics['brier_score']:.4f}")

    print(
        f"Mean predicted: {probabilities.mean():.2%}"
    )
    print(f"Mean observed:  {y.mean():.2%}")

    print(
        f"Highest/lowest risk ratio: {risk_ratio:.2f}x"
    )
    print(
        f"Charge-offs captured in highest decile: "
        f"{top_decile_capture:.2f}%"
    )

    print("\nRisk deciles:")
    print(
        summary[
            ["decile", "loans", "predicted_pct", "observed_pct"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    filename = (
        name.lower().replace(" ", "_")
        + "_deciles.csv"
    )

    summary.to_csv(REPORT_DIR / filename, index=False)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(
        DATA_DIR / "loan_validation.csv",
        usecols=FEATURES + ["target"],
    )

    X = validation[FEATURES]
    y = validation["target"].astype(int)

    for name, filename in MODELS.items():
        model = joblib.load(DATA_DIR / filename)
        analyse_model(name, model, X, y)

    print("\nBoth models evaluated on 2014 validation only.")
    print("2015 test data remains untouched.")


if __name__ == "__main__":
    main()
