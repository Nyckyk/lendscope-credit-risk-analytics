from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/raw/accepted_2007_to_2018Q4.csv"
)

EXPECTED_COLUMN_COUNT = 151

REQUIRED_COLUMNS = {
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
}


def read_source_columns() -> list[str]:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_PATH}"
        )

    header = pd.read_csv(
        SOURCE_PATH,
        nrows=0,
    )

    return header.columns.tolist()


def validate_source_schema() -> list[str]:
    columns = read_source_columns()

    errors = []

    if len(columns) != EXPECTED_COLUMN_COUNT:
        errors.append(
            "Expected "
            f"{EXPECTED_COLUMN_COUNT} columns, "
            f"found {len(columns)}."
        )

    missing_required = (
        REQUIRED_COLUMNS - set(columns)
    )

    if missing_required:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_required))
        )

    duplicated_columns = (
        pd.Index(columns)
        .duplicated()
    )

    if duplicated_columns.any():
        duplicates = [
            column
            for column, duplicated
            in zip(columns, duplicated_columns)
            if duplicated
        ]

        errors.append(
            "Duplicate columns found: "
            + ", ".join(duplicates)
        )

    if errors:
        error_text = "\n".join(
            f"- {error}" for error in errors
        )

        raise ValueError(
            "Source schema validation failed:\n"
            f"{error_text}"
        )

    print("Source schema validation passed")
    print(f"Column count: {len(columns)}")
    print(
        "Required columns present: "
        f"{len(REQUIRED_COLUMNS)}"
    )

    return columns


if __name__ == "__main__":
    validate_source_schema()