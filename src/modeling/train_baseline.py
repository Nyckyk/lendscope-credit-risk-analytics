
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

NUMERIC_FEATURES = [
    "loan_amount",
    "annual_income",
    "debt_to_income_ratio",
    "fico_range_low",
]

CATEGORICAL_FEATURES = [
    "loan_purpose",
    "home_ownership",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(filename):
    path = DATA_DIR / filename

    df = pd.read_csv(
        path,
        usecols=FEATURES + ["target"],
    )

    X = df[FEATURES]
    y = df["target"].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Unexpected target values")

    return X, y


def make_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    return model


def evaluate(y_true, probabilities):
    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
    }


def main():
    print("Loading training data...")
    X_train, y_train = load_dataset("loan_train.csv")

    print("Loading validation data...")
    X_val, y_val = load_dataset("loan_validation.csv")

    print(f"Training loans: {len(X_train):,}")
    print(f"Validation loans: {len(X_val):,}")

    # Constant-probability baseline.
    training_prevalence = float(y_train.mean())

    baseline_probs = np.full(
        len(y_val),
        training_prevalence,
    )

    baseline_metrics = evaluate(
        y_val,
        baseline_probs,
    )

    print("\nTraining Logistic Regression...")

    model = make_pipeline()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    model_metrics = evaluate(
        y_val,
        probabilities,
    )

    print("\n=== VALIDATION RESULTS ===")

    for name, metrics in [
        ("Constant baseline", baseline_metrics),
        ("Logistic Regression", model_metrics),
    ]:
        print(f"\n{name}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Brier score:       {metrics['brier_score']:.4f}")

    # Save fitted preprocessing and model together.
    model_path = DATA_DIR / "logistic_baseline.joblib"
    joblib.dump(model, model_path)

    reports_dir = PROJECT_ROOT / "reports" / "modeling"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "training_loans": int(len(y_train)),
        "validation_loans": int(len(y_val)),
        "training_chargeoff_share": training_prevalence,
        "validation_chargeoff_share": float(y_val.mean()),
        "features": FEATURES,
        "constant_baseline": baseline_metrics,
        "logistic_regression": model_metrics,
        "test_set_evaluated": False,
    }

    report_path = reports_dir / "logistic_baseline_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
