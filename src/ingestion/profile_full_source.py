from collections import Counter
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
CHUNK_SIZE = 50_000


def profile_full_source() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Source file not found: {DATA_PATH}"
        )

    total_rows = 0
    missing_counts = None
    loan_status_counts = Counter()

    min_issue_date = None
    max_issue_date = None

    print(f"Profiling: {DATA_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print()

    reader = pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk

        chunk_missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts = missing_counts.add(
                chunk_missing,
                fill_value=0,
            )

        if "loan_status" in chunk.columns:
            status_counts = (
                chunk["loan_status"]
                .dropna()
                .value_counts()
            )

            loan_status_counts.update(
                status_counts.to_dict()
            )

        if "issue_d" in chunk.columns:
            parsed_dates = pd.to_datetime(
                chunk["issue_d"],
                format="%b-%Y",
                errors="coerce",
            )

            chunk_min = parsed_dates.min()
            chunk_max = parsed_dates.max()

            if pd.notna(chunk_min):
                if min_issue_date is None or chunk_min < min_issue_date:
                    min_issue_date = chunk_min

            if pd.notna(chunk_max):
                if max_issue_date is None or chunk_max > max_issue_date:
                    max_issue_date = chunk_max

        print(
            f"Chunk {chunk_number:3}: "
            f"{rows_in_chunk:,} rows | "
            f"Total processed: {total_rows:,}"
        )

    print()
    print("=== FULL DATASET ===")
    print(f"Total rows: {total_rows:,}")

    print()
    print("=== ISSUE DATE RANGE ===")
    print(f"Earliest issue date: {min_issue_date}")
    print(f"Latest issue date:   {max_issue_date}")

    print()
    print("=== LOAN STATUS COUNTS ===")

    for status, count in loan_status_counts.most_common():
        percentage = count / total_rows * 100

        print(
            f"{status:<25} "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("=== HIGHEST MISSING VALUE RATES ===")

    missing_percentage = (
        missing_counts
        .div(total_rows)
        .mul(100)
        .sort_values(ascending=False)
    )

    print(
        missing_percentage
        .head(30)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    profile_full_source()