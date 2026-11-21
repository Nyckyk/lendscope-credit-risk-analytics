
import json

import joblib

from src.modeling.train_expanded_model import (
    DATA_DIR,
    REPORT_DIR,
    FEATURES,
    load_expanded,
    make_model,
)

from src.modeling.train_baseline import evaluate


REMOVED_FEATURES = [
    "delinq_2yrs",
    "verification_status",
]

REDUCED_FEATURES = [
    feature for feature in FEATURES
    if feature not in REMOVED_FEATURES
]


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )
    print(f"Brier score:       {metrics['brier_score']:.4f}")


def make_reduced_model():
    # Start with the same model configuration as the
    # original 14-feature Gradient Boosting experiment.
    model = make_model()

    preprocessor = model.named_steps["preprocessor"]

    # Remove the two features from the preprocessing
    # pipeline without changing any model hyperparameters.
    updated_transformers = []

    for name, transformer, columns in preprocessor.transformers:
        selected_columns = [
            column for column in columns
            if column in REDUCED_FEATURES
        ]

        updated_transformers.append(
            (name, transformer, selected_columns)
        )

    preprocessor.set_params(
        transformers=updated_transformers
    )

    return model


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading modelling datasets...", flush=True)

    train = load_expanded("train")
    validation = load_expanded("validation")

    X_train = train[REDUCED_FEATURES]
    y_train = train["target"].astype(int)

    X_val = validation[REDUCED_FEATURES]
    y_val = validation["target"].astype(int)

    print(f"Training loans: {len(train):,}")
    print(f"Validation loans: {len(validation):,}")

    # Existing trained 14-feature model.
    print("\nEvaluating original model...", flush=True)

    original_model = joblib.load(
        DATA_DIR / "hist_gradient_boosting_expanded.joblib"
    )

    original_probabilities = original_model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    original_metrics = evaluate(
        y_val,
        original_probabilities,
    )

    # New 12-feature model.
    print("\nTraining reduced model...", flush=True)

    reduced_model = make_reduced_model()

    reduced_model.fit(
        X_train,
        y_train,
    )

    reduced_probabilities = reduced_model.predict_proba(
        X_val
    )[:, 1]

    reduced_metrics = evaluate(
        y_val,
        reduced_probabilities,
    )

    print("\n=== FEATURE ABLATION RESULTS ===")

    print_metrics(
        "Original Gradient Boosting (14 features)",
        original_metrics,
    )

    print_metrics(
        "Reduced Gradient Boosting (12 features)",
        reduced_metrics,
    )

    print("\n=== DIFFERENCE: REDUCED MINUS ORIGINAL ===")

    for metric in original_metrics:
        difference = (
            reduced_metrics[metric]
            - original_metrics[metric]
        )

        print(f"{metric}: {difference:+.5f}")

    # Save the candidate and reproducible results.
    model_path = (
        DATA_DIR / "hist_gradient_boosting_reduced.joblib"
    )

    joblib.dump(
        reduced_model,
        model_path,
    )

    report = {
        "training_years": "2011-2013",
        "validation_year": 2014,
        "original_feature_count": len(FEATURES),
        "reduced_feature_count": len(REDUCED_FEATURES),
        "removed_features": REMOVED_FEATURES,
        "reduced_features": REDUCED_FEATURES,
        "original_metrics": original_metrics,
        "reduced_metrics": reduced_metrics,
        "test_set_evaluated": False,
    }

    report_path = (
        REPORT_DIR / "feature_ablation_validation.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")
    print("2015 test data remains untouched.")


if __name__ == "__main__":
    main()
