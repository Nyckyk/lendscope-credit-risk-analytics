
from pathlib import Path
from time import perf_counter

import pandas as pd
from sqlalchemy import text

from src.ingestion.database import get_engine


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"

CHUNK_SIZE = 25_000

# Verified using the earlier SQL cohort audits.
EXPECTED = {
    "train": (157_993, 19_780),
    "validation": (162_570, 22_315),
}

NEW_NUMERIC = [
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
    "open_acc",
]


# ============================================================
# SQL QUERY
# ============================================================

# Read directly from raw.loans to avoid joining the large
# staging view back to the raw table.
#
# Only use 2011-2014 loans:
#   2011-2013 = training
#   2014      = validation
#
# The 2015 test cohort is deliberately excluded.
#
# Filtering issue_d using its Mon-YYYY text format avoids
# repeatedly converting dates across the entire raw table.

QUERY = """
SELECT
    CASE
        WHEN RIGHT(BTRIM(r.issue_d), 4)
             IN ('2011', '2012', '2013')
        THEN 'train'
        ELSE 'validation'
    END AS dataset_split,

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

  AND BTRIM(r.issue_d)
      ~* '^[a-z]{3}-201[1-4]$'

  AND r.loan_status IN (
      'Fully Paid',
      'Charged Off'
  )
"""


# ============================================================
# DATA TRANSFORMATIONS
# ============================================================

def transform_chunk(chunk):
    """Clean and engineer features within one batch."""

    chunk = chunk.copy()

    # Convert numeric fields stored as text.
    for column in NEW_NUMERIC:

        original = chunk[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        # Detect unexpected nonempty, nonnumeric values.
        invalid = (
            original.notna()
            & original.astype(str).str.strip().ne("")
            & converted.isna()
        )

        if invalid.any():
            examples = (
                original[invalid]
                .astype(str)
                .unique()[:5]
                .tolist()
            )

            raise ValueError(
                f"Invalid values in {column}: {examples}"
            )

        chunk[column] = converted

    # Missing employment history should not be interpreted
    # as zero years of employment.
    chunk["employment_length"] = (
        chunk["employment_length"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown")
    )

    # Treat missing verification status explicitly.
    chunk["verification_status"] = (
        chunk["verification_status"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown")
    )

    # Convert earliest credit line dates.
    # Verified source format: Mon-YYYY.
    credit_start = pd.to_datetime(
        chunk["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    issue_date = pd.to_datetime(
        chunk["issue_date"],
        errors="coerce",
    )

    if credit_start.isna().any():
        raise ValueError(
            "Invalid earliest credit line dates detected"
        )

    if issue_date.isna().any():
        raise ValueError(
            "Invalid loan issue dates detected"
        )

    # Calculate the length of credit history at origination.
    credit_age_months = (
        (issue_date.dt.year - credit_start.dt.year) * 12
        + issue_date.dt.month
        - credit_start.dt.month
    )

    if (credit_age_months < 0).any():
        raise ValueError(
            "Credit history begins after loan origination"
        )

    chunk["credit_history_months"] = (
        credit_age_months.astype("int32")
    )

    # Remove the original credit-history date.
    chunk = chunk.drop(
        columns=["earliest_cr_line"]
    )

    # Verify the target is strictly binary.
    if not chunk["target"].isin([0, 1]).all():
        raise ValueError(
            "Unexpected target values detected"
        )

    return chunk


# ============================================================
# DATASET EXTRACTION
# ============================================================

def main():

    start_time = perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Final output files.
    paths = {
        split: OUTPUT_DIR / f"loan_{split}_expanded.csv"
        for split in EXPECTED
    }

    # Temporary outputs prevent failed runs from replacing
    # the final validated datasets.
    temp_paths = {
        split: OUTPUT_DIR / f"loan_{split}_expanded.csv.tmp"
        for split in EXPECTED
    }

    stats = {
        split: {
            "rows": 0,
            "charged": 0,
            "missing_revol": 0,
            "util_over_100": 0,
            "missing_employment": 0,
        }
        for split in EXPECTED
    }

    # Remove temporary files from previous interrupted runs.
    for path in temp_paths.values():
        path.unlink(missing_ok=True)

    print("=" * 55, flush=True)
    print("LENDSCOPE: EXPANDED MODELLING DATASET", flush=True)
    print("=" * 55, flush=True)

    print(
        "\nConnecting to PostgreSQL...",
        flush=True,
    )

    engine = get_engine()

    try:

        with engine.connect() as connection:

            # Request server-side streaming where supported
            # to avoid loading the whole result into memory.
            connection = connection.execution_options(
                stream_results=True,
            )

            print(
                "Executing SQL and fetching loan batches...",
                flush=True,
            )

            chunks = pd.read_sql_query(
                text(QUERY),
                connection,
                chunksize=CHUNK_SIZE,
            )

            for batch_number, chunk in enumerate(
                chunks,
                start=1,
            ):

                # Apply feature engineering.
                chunk = transform_chunk(chunk)

                # Separate training and validation loans.
                for split, group in chunk.groupby(
                    "dataset_split",
                    sort=False,
                ):

                    if split not in EXPECTED:
                        raise ValueError(
                            f"Unexpected split: {split}"
                        )

                    output = group.drop(
                        columns=["dataset_split"]
                    )

                    # Update validation counters.
                    stats[split]["rows"] += len(output)

                    stats[split]["charged"] += int(
                        output["target"].sum()
                    )

                    stats[split]["missing_revol"] += int(
                        output["revol_util"].isna().sum()
                    )

                    stats[split]["util_over_100"] += int(
                        (output["revol_util"] > 100).sum()
                    )

                    stats[split]["missing_employment"] += int(
                        (
                            output["employment_length"]
                            == "Unknown"
                        ).sum()
                    )

                    # Write each batch to its temporary CSV.
                    path = temp_paths[split]

                    output.to_csv(
                        path,
                        mode="a",
                        header=not path.exists(),
                        index=False,
                    )

                # Display progress after every batch.
                processed = sum(
                    values["rows"]
                    for values in stats.values()
                )

                elapsed = perf_counter() - start_time

                print(
                    f"Batch {batch_number:02d} | "
                    f"{processed:,} loans processed | "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print(
            "\n=== EXPANDED DATASET VALIDATION ===",
            flush=True,
        )

        for split, (
            expected_rows,
            expected_charged,
        ) in EXPECTED.items():

            result = stats[split]

            print(f"\n{split.upper()}", flush=True)

            print(
                f"Rows: {result['rows']:,}",
                flush=True,
            )

            print(
                f"Charged off: {result['charged']:,}",
                flush=True,
            )

            print(
                "Missing revolving utilisation: "
                f"{result['missing_revol']:,}",
                flush=True,
            )

            print(
                "Revolving utilisation over 100%: "
                f"{result['util_over_100']:,}",
                flush=True,
            )

            print(
                "Unknown employment length: "
                f"{result['missing_employment']:,}",
                flush=True,
            )

            if result["rows"] != expected_rows:
                raise RuntimeError(
                    f"{split}: Expected {expected_rows:,} rows, "
                    f"got {result['rows']:,}"
                )

            if result["charged"] != expected_charged:
                raise RuntimeError(
                    f"{split}: Expected "
                    f"{expected_charged:,} charge-offs, "
                    f"got {result['charged']:,}"
                )

            if result["charged"] > result["rows"]:
                raise RuntimeError(
                    f"{split}: Invalid target count"
                )

            if not temp_paths[split].exists():
                raise RuntimeError(
                    f"{split}: Output file was not created"
                )

        # ====================================================
        # PUBLISH VALIDATED DATASETS
        # ====================================================

        # Only replace final outputs after all checks pass.
        for split in EXPECTED:

            temp_paths[split].replace(
                paths[split]
            )

            print(
                f"\nSaved: {paths[split]}",
                flush=True,
            )

        total_elapsed = perf_counter() - start_time

        print("\n" + "=" * 55)
        print("ALL EXPANDED DATASETS VALIDATED")
        print("=" * 55)

        print(
            f"Total loans: "
            f"{sum(s['rows'] for s in stats.values()):,}"
        )

        print(
            f"Elapsed time: {total_elapsed:.1f} seconds"
        )

        print(
            "The 2015 test dataset was not accessed."
        )

    except Exception:

        # Do not leave misleading temporary datasets behind
        # if extraction or validation fails.
        for path in temp_paths.values():
            path.unlink(missing_ok=True)

        raise


if __name__ == "__main__":
    main()
