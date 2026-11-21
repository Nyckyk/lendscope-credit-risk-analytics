
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.modeling.train_baseline import (
    PROJECT_ROOT,
    DATA_DIR,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    load_dataset,
    evaluate,
)


REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def make_gradient_boosting_pipeline():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
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
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
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


def main():

    print("Loading chronological datasets...")

    X_train, y_train = load_dataset(
        "loan_train.csv"
    )

    X_val, y_val = load_dataset(
        "loan_validation.csv"
    )

    print(f"Training: {len(X_train):,} loans")
    print(f"Validation: {len(X_val):,} loans")

    print("\nLoading Logistic Regression baseline...")

    baseline = joblib.load(
        DATA_DIR / "logistic_baseline.joblib"
    )

    baseline_probs = baseline.predict_proba(
        X_val
    )[:, 1]

    print("\nTraining Histogram Gradient Boosting...")

    candidate = make_gradient_boosting_pipeline()

    candidate.fit(X_train, y_train)

    candidate_probs = candidate.predict_proba(
        X_val
    )[:, 1]

    baseline_metrics = evaluate(
        y_val,
        baseline_probs,
    )

    candidate_metrics = evaluate(
        y_val,
        candidate_probs,
    )

    print("\n=== MODEL COMPARISON: 2014 VALIDATION ===")

    for name, metrics in [
        ("Logistic Regression", baseline_metrics),
        ("Histogram Gradient Boosting", candidate_metrics),
    ]:

        print(f"\n{name}")

        print(
            f"ROC-AUC:           "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.4f}"
        )

        print(
            f"Brier score:       "
            f"{metrics['brier_score']:.4f}"
        )

    # Save candidate for later analysis.
    model_path = (
        DATA_DIR / "hist_gradient_boosting.joblib"
    )

    joblib.dump(candidate, model_path)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "training_years": "2011-2013",
        "validation_year": 2014,
        "training_loans": len(y_train),
        "validation_loans": len(y_val),
        "features": FEATURES,
        "logistic_regression": baseline_metrics,
        "hist_gradient_boosting": candidate_metrics,
        "test_set_evaluated": False,
    }

    report_path = (
        REPORT_DIR / "model_comparison_validation.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nCandidate saved: {model_path}")
    print(f"Comparison report: {report_path}")

    print("\nThe 2015 test set has not been used.")


if __name__ == "__main__":
    main()
