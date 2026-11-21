
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def main():
    # Read the exact feature list used during training.
    training_report = json.loads(
        (REPORT_DIR / "logistic_baseline_validation.json")
        .read_text(encoding="utf-8")
    )

    features = training_report["features"]

    # Load the saved preprocessing and Logistic Regression model.
    model = joblib.load(
        DATA_DIR / "logistic_baseline.joblib"
    )

    # Only use validation data. Leave the test dataset untouched.
    validation = pd.read_csv(
        DATA_DIR / "loan_validation.csv",
        usecols=features + ["target"],
    )

    X = validation[features]
    y = validation["target"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "actual": y.to_numpy(),
        "predicted": probabilities,
    })

    # Divide loans into ten equally sized groups by predicted risk.
    # Decile 1 = lowest predicted risk.
    # Decile 10 = highest predicted risk.
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

    print("\n=== MODEL CALIBRATION ===")
    print(
        f"Average predicted charge-off: "
        f"{probabilities.mean():.2%}"
    )
    print(
        f"Actual validation charge-off: "
        f"{y.mean():.2%}"
    )

    print("\n=== RISK DECILES ===")

    display_columns = [
        "decile",
        "loans",
        "charged_off",
        "predicted_pct",
        "observed_pct",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    lowest = summary.iloc[0]["observed_rate"]
    highest = summary.iloc[-1]["observed_rate"]

    if lowest > 0:
        print(
            f"\nHighest/lowest decile risk ratio: "
            f"{highest / lowest:.2f}x"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "validation_risk_deciles.csv"

    summary[display_columns].to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
