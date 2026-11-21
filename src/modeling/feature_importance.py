
from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.modeling.train_expanded_model import (
    DATA_DIR,
    REPORT_DIR,
    FEATURES,
)


MODEL_PATH = (
    DATA_DIR / "hist_gradient_boosting_expanded.joblib"
)

VALIDATION_PATH = (
    DATA_DIR / "loan_validation_expanded.csv"
)

SAMPLE_SIZE = 20_000
N_REPEATS = 3
RANDOM_STATE = 42


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading expanded model...", flush=True)

    model = joblib.load(MODEL_PATH)

    print("Loading 2014 validation data...", flush=True)

    df = pd.read_csv(
        VALIDATION_PATH,
        usecols=FEATURES + ["target"],
    )

    if not df["target"].isin([0, 1]).all():
        raise ValueError("Invalid target values")

    if len(df) < SAMPLE_SIZE:
        raise ValueError("Insufficient validation rows")

    # Stratified sampling preserves approximately the
    # same charge-off proportion as the validation cohort.
    sample, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )

    X = sample[FEATURES]
    y = sample["target"].astype(int)

    print(f"Sample size: {len(sample):,}")
    print(f"Observed charge-off share: {y.mean():.2%}")

    predictions = model.predict_proba(X)[:, 1]

    reference_auc = roc_auc_score(y, predictions)

    print(
        f"Reference sample ROC-AUC: {reference_auc:.4f}",
        flush=True,
    )

    print(
        "\nCalculating permutation importance...",
        flush=True,
    )

    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_auc_drop": result.importances_mean,
        "std_auc_drop": result.importances_std,
    })

    importance = importance.sort_values(
        "mean_auc_drop",
        ascending=False,
    ).reset_index(drop=True)

    print("\n=== FEATURE IMPORTANCE ===")

    print(
        importance.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    output_path = (
        REPORT_DIR / "expanded_feature_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")
    print("2015 test data was not accessed.")


if __name__ == "__main__":
    main()
