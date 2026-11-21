
from pathlib import Path
from time import perf_counter

import pandas as pd
from sqlalchemy import text

from src.ingestion.database import get_engine
from src.modeling.build_expanded_dataset import (
    CHUNK_SIZE,
    OUTPUT_DIR,
    transform_chunk,
)


EXPECTED_ROWS = 283_026
EXPECTED_CHARGED_OFF = 42_132

ORIGINAL_PATH = OUTPUT_DIR / "loan_test.csv"
OUTPUT_PATH = OUTPUT_DIR / "loan_test_expanded.csv"
TEMP_PATH = OUTPUT_DIR / "loan_test_expanded.csv.tmp"


# Use the same source fields and conversions as the
# expanded training and validation datasets.
QUERY = """
SELECT
    r.id::BIGINT AS loan_id,

    TO_DATE(
        BTRIM(r.issue_d),
        'Mon-YYYY'
    ) AS issue_date,

    NULLIF(BTRIM(r.loan_amnt), '')::NUMERIC
        AS loan_amount,

    NULLIF(BTRIM(r.annual_inc), '')::NUMERIC
        AS annual_income,

    NULLIF(BTRIM(r.dti), '')::NUMERIC
        AS debt_to_income_ratio,

    NULLIF(BTRIM(r.fico_range_low), '')::NUMERIC::INTEGER
        AS fico_range_low,

    NULLIF(BTRIM(r.fico_range_high), '')::NUMERIC::INTEGER
        AS fico_range_high,

    r.purpose AS loan_purpose,
    r.home_ownership,

    r.emp_length AS employment_length,
    r.verification_status,

    r.revol_util,
    r.inq_last_6mths,
    r.delinq_2yrs,
    r.pub_rec,
    r.open_acc,

    r.earliest_cr_line,

    CASE
        WHEN r.loan_status = 'Charged Off'
        THEN 1
        ELSE 0
    END AS target

FROM raw.loans AS r

WHERE r.id ~ '^[0-9]+$'
  AND BTRIM(r.term) = '36 months'
  AND BTRIM(r.issue_d) ~* '^[a-z]{3}-2015$'
  AND r.loan_status IN (
      'Fully Paid',
      'Charged Off'
  )
"""


def read_id_targets(path):
    df = pd.read_csv(
        path,
        usecols=["loan_id", "target"],
    )

    if df["loan_id"].isna().any():
        raise ValueError(f"Missing IDs in {path.name}")

    if not df["loan_id"].is_unique:
        raise ValueError(f"Duplicate IDs in {path.name}")

    if not df["target"].isin([0, 1]).all():
        raise ValueError(f"Invalid targets in {path.name}")

    return (
        df.set_index("loan_id")["target"]
        .sort_index()
    )


def main():
    start = perf_counter()

    if not ORIGINAL_PATH.is_file():
        raise FileNotFoundError(
            f"Original test dataset missing: {ORIGINAL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.unlink(missing_ok=True)

    rows = 0
    charged_off = 0
    missing_revol = 0

    print("Building expanded 2015 test dataset...", flush=True)

    try:
        engine = get_engine()

        with engine.connect() as connection:
            connection = connection.execution_options(
                stream_results=True,
            )

            chunks = pd.read_sql_query(
                text(QUERY),
                connection,
                chunksize=CHUNK_SIZE,
            )

            for batch_number, chunk in enumerate(
                chunks, start=1
            ):
                chunk = transform_chunk(chunk)

                issue_years = pd.to_datetime(
                    chunk["issue_date"]
                ).dt.year

                if not issue_years.eq(2015).all():
                    raise ValueError(
                        "Unexpected issue year in test data"
                    )

                rows += len(chunk)
                charged_off += int(chunk["target"].sum())
                missing_revol += int(
                    chunk["revol_util"].isna().sum()
                )

                chunk.to_csv(
                    TEMP_PATH,
                    mode="a",
                    header=not TEMP_PATH.exists(),
                    index=False,
                )

                print(
                    f"Batch {batch_number:02d}: "
                    f"{rows:,} records extracted",
                    flush=True,
                )

        print("\n=== TEST DATASET CHECKS ===")

        print(f"Rows: {rows:,}")
        print(f"Charged off: {charged_off:,}")
        print(f"Missing revolving utilisation: {missing_revol:,}")

        if rows != EXPECTED_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS:,} loans, "
                f"received {rows:,}"
            )

        if charged_off != EXPECTED_CHARGED_OFF:
            raise RuntimeError(
                f"Expected {EXPECTED_CHARGED_OFF:,} charge-offs, "
                f"received {charged_off:,}"
            )

        print(
            "\nChecking original and expanded loan identities...",
            flush=True,
        )

        original = read_id_targets(ORIGINAL_PATH)
        expanded = read_id_targets(TEMP_PATH)

        if not original.equals(expanded):
            raise RuntimeError(
                "Expanded test loan IDs or targets "
                "do not match the original test dataset"
            )

        # Publish only after every validation passes.
        TEMP_PATH.replace(OUTPUT_PATH)

        print("\nALL TEST DATASET CHECKS PASSED")
        print("Loan IDs and outcomes match exactly.")
        print(f"Saved: {OUTPUT_PATH}")
        print(
            f"Elapsed: {perf_counter() - start:.1f} seconds"
        )
        print("\nNo model predictions were generated.")

    finally:
        TEMP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
