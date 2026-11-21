
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.modeling.train_baseline import (
    DATA_DIR,
    PROJECT_ROOT,
    FEATURES as BASE_FEATURES,
    evaluate,
)


REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

# Original six features.
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

# Expanded features.
NUMERIC_FEATURES += [
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
    "open_acc",
    "credit_history_months",
]

CATEGORICAL_FEATURES += [
    "employment_length",
    "verification_status",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPECTED = {
    "train": (157_993, 19_780),
    "validation": (162_570, 22_315),
}


def load_expanded(split):
    path = DATA_DIR / f"loan_{split}_expanded.csv"

    df = pd.read_csv(
        path,
        usecols=["loan_id", "target"] + FEATURES,
    )

    expected_rows, expected_charged = EXPECTED[split]

    if len(df) != expected_rows:
        raise ValueError(f"{split}: unexpected row count")

    if int(df["target"].sum()) != expected_charged:
        raise ValueError(f"{split}: unexpected charge-off count")

    if df["loan_id"].isna().any():
        raise ValueError(f"{split}: missing loan IDs")

    if not df["loan_id"].is_unique:
        raise ValueError(f"{split}: duplicate loan IDs")

    if not df["target"].isin([0, 1]).all():
        raise ValueError(f"{split}: invalid target")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


def verify_original_cohort(split, expanded):
    original_path = DATA_DIR / f"loan_{split}.csv"

    original = pd.read_csv(
        original_path,
        usecols=["loan_id", "target"],
    )

    if not original["loan_id"].is_unique:
        raise ValueError("Original dataset has duplicate IDs")

    expected = (
        original.set_index("loan_id")["target"]
        .sort_index()
    )

    actual = (
        expanded.set_index("loan_id")["target"]
        .sort_index()
    )

    if not expected.equals(actual):
        raise ValueError(
            f"{split}: expanded cohort does not match "
            "the original dataset"
        )

    print(f"{split}: loan identities and targets match.")


def make_model():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def risk_deciles(y, probabilities):
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
        summary["predicted_rate"] * 100
    )

    summary["observed_pct"] = (
        summary["observed_rate"] * 100
    )

    return summary


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading expanded datasets...", flush=True)

    train = load_expanded("train")
    validation = load_expanded("validation")

    # Check that our new SQL extraction selected exactly
    # the same loan IDs and outcomes as the original data.
    verify_original_cohort("train", train)
    verify_original_cohort("validation", validation)

    X_train = train[FEATURES]
    y_train = train["target"].astype(int)

    X_val = validation[FEATURES]
    y_val = validation["target"].astype(int)

    print(f"Training loans: {len(train):,}")
    print(f"Validation loans: {len(validation):,}")
    print(f"Features: {len(FEATURES)}")

    # ----------------------------------------------------
    # Existing six-feature Gradient Boosting
    # ----------------------------------------------------

    print("\nEvaluating existing model...", flush=True)

    original_validation = pd.read_csv(
        DATA_DIR / "loan_validation.csv",
        usecols=["loan_id", "target"] + BASE_FEATURES,
    )

    # Align original data to the same loan ordering.
    original_validation = (
        original_validation
        .set_index("loan_id")
        .loc[validation["loan_id"]]
    )

    existing_model = joblib.load(
        DATA_DIR / "hist_gradient_boosting.joblib"
    )

    existing_probs = existing_model.predict_proba(
        original_validation[BASE_FEATURES]
    )[:, 1]

    existing_metrics = evaluate(
        y_val,
        existing_probs,
    )

    # ----------------------------------------------------
    # Expanded 14-feature Gradient Boosting
    # ----------------------------------------------------

    print("\nTraining expanded model...", flush=True)

    expanded_model = make_model()

    expanded_model.fit(
        X_train,
        y_train,
    )

    print("Evaluating expanded model...", flush=True)

    expanded_probs = expanded_model.predict_proba(
        X_val
    )[:, 1]

    expanded_metrics = evaluate(
        y_val,
        expanded_probs,
    )

    print("\n=== VALIDATION MODEL COMPARISON ===")

    for name, metrics in [
        ("Existing Gradient Boosting", existing_metrics),
        ("Expanded Gradient Boosting", expanded_metrics),
    ]:
        print(f"\n{name}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.4f}"
        )
        print(
            f"Brier score:       "
            f"{metrics['brier_score']:.4f}"
        )

    # ----------------------------------------------------
    # Risk deciles
    # ----------------------------------------------------

    summary = risk_deciles(
        y_val,
        expanded_probs,
    )

    print("\n=== EXPANDED MODEL RISK DECILES ===")

    print(
        summary[
            ["decile", "loans", "predicted_pct", "observed_pct"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print(
        f"\nMean predicted charge-off: "
        f"{expanded_probs.mean():.2%}"
    )

    print(
        f"Actual charge-off: "
        f"{y_val.mean():.2%}"
    )

    lowest = summary.iloc[0]["observed_rate"]
    highest = summary.iloc[-1]["observed_rate"]

    if lowest > 0:
        print(
            f"Highest/lowest risk ratio: "
            f"{highest / lowest:.2f}x"
        )

    top_capture = (
        summary.iloc[-1]["charged_off"]
        / y_val.sum() * 100
    )

    print(
        f"Charge-offs captured in top decile: "
        f"{top_capture:.2f}%"
    )

    # ----------------------------------------------------
    # Save experiment
    # ----------------------------------------------------

    model_path = (
        DATA_DIR / "hist_gradient_boosting_expanded.joblib"
    )

    joblib.dump(
        expanded_model,
        model_path,
    )

    summary.to_csv(
        REPORT_DIR / "expanded_model_deciles.csv",
        index=False,
    )

    report = {
        "training_years": "2011-2013",
        "validation_year": 2014,
        "training_loans": len(train),
        "validation_loans": len(validation),
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "sklearn_version": sklearn_version,
        "existing_model": existing_metrics,
        "expanded_model": expanded_metrics,
        "test_set_evaluated": False,
    }

    report_path = (
        REPORT_DIR / "expanded_model_validation.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")

    print("\n2015 test data remains untouched.")


if __name__ == "__main__":
    main()
