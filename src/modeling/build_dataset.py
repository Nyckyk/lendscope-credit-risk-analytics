
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.ingestion.database import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_SIZE = 25_000

FEATURES = [
    "loan_amount",
    "annual_income",
    "debt_to_income_ratio",
    "fico_range_low",
    "fico_range_high",
    "loan_purpose",
    "home_ownership",
]

EXPECTED = {
    "train": (157_993, 19_780),
    "validation": (162_570, 22_315),
    "test": (283_026, 42_132),
}

QUERY = """
SELECT
    CASE
        WHEN issue_date < DATE '2014-01-01'
            THEN 'train'
        WHEN issue_date < DATE '2015-01-01'
            THEN 'validation'
        ELSE 'test'
    END AS dataset_split,

    loan_id,
    issue_date,

    loan_amount,
    annual_income,
    debt_to_income_ratio,
    fico_range_low,
    fico_range_high,
    loan_purpose,
    home_ownership,

    CASE
        WHEN loan_status = 'Charged Off' THEN 1
        ELSE 0
    END AS target

FROM staging.stg_loans

WHERE term_months = 36
  AND issue_date >= DATE '2011-01-01'
  AND issue_date < DATE '2016-01-01'
  AND loan_status IN ('Fully Paid', 'Charged Off')
"""


def build_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        name: {"rows": 0, "charged": 0, "missing": 0}
        for name in EXPECTED
    }

    paths = {
        name: OUTPUT_DIR / f"loan_{name}.csv"
        for name in EXPECTED
    }

    # Remove outputs from previous executions.
    for path in paths.values():
        if path.exists():
            path.unlink()

    engine = get_engine()

    with engine.connect() as connection:
        connection = connection.execution_options(
            stream_results=True
        )

        chunks = pd.read_sql_query(
            text(QUERY),
            connection,
            chunksize=CHUNK_SIZE,
        )

        for chunk in chunks:
            for name, group in chunk.groupby(
                "dataset_split", sort=False
            ):
                if name not in stats:
                    raise ValueError(f"Unexpected split: {name}")

                if not group["target"].isin([0, 1]).all():
                    raise ValueError("Invalid target value")

                output = group.drop(columns="dataset_split")

                output.to_csv(
                    paths[name],
                    mode="a",
                    header=not paths[name].exists(),
                    index=False,
                )

                stats[name]["rows"] += len(output)

                stats[name]["charged"] += int(
                    output["target"].sum()
                )

                stats[name]["missing"] += int(
                    output[FEATURES].isna().any(axis=1).sum()
                )

    print("\n=== MODEL DATASET VALIDATION ===")

    for name, (expected_rows, expected_charged) in EXPECTED.items():
        result = stats[name]

        print(f"\n{name.upper()}")
        print(f"Rows: {result['rows']:,}")
        print(f"Charged off: {result['charged']:,}")
        print(f"Rows with missing features: {result['missing']:,}")

        if result["rows"] != expected_rows:
            raise RuntimeError(
                f"{name}: unexpected row count"
            )

        if result["charged"] != expected_charged:
            raise RuntimeError(
                f"{name}: unexpected target count"
            )

    print("\nAll modelling datasets validated.")


if __name__ == "__main__":
    build_dataset()
