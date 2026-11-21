from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
SAMPLE_ROWS = 10_000


def profile_source() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Source file not found: {DATA_PATH}"
        )

    file_size_gb = DATA_PATH.stat().st_size / (1024 ** 3)

    print(f"Source file: {DATA_PATH}")
    print(f"File size: {file_size_gb:.2f} GiB")

    sample = pd.read_csv(
        DATA_PATH,
        nrows=SAMPLE_ROWS,
        low_memory=False,
    )

    print()
    print("=== DATASET SAMPLE ===")
    print(f"Rows sampled: {len(sample):,}")
    print(f"Number of columns: {len(sample.columns):,}")

    print()
    print("=== COLUMN NAMES ===")

    for index, column in enumerate(sample.columns, start=1):
        print(f"{index:3}. {column}")

    print()
    print("=== INFERRED DATA TYPES ===")
    print(sample.dtypes.to_string())

    print()
    print("=== HIGHEST MISSING VALUE RATES ===")

    missing_percentage = (
        sample.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(missing_percentage.head(20).round(2).to_string())

    print()
    print("=== LOAN STATUS VALUES ===")

    if "loan_status" in sample.columns:
        print(
            sample["loan_status"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("loan_status column not found")


if __name__ == "__main__":
    profile_source()