
from sqlalchemy import text

from src.ingestion.database import get_engine


EXPECTED_TOTAL_ROWS = 2_260_701
EXPECTED_LOAN_ROWS = 2_260_668
EXPECTED_METADATA_ROWS = 33
EXPECTED_COLUMNS = 151


def check_raw_loans() -> None:
    engine = get_engine()

    query = text("""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(*) FILTER (
                WHERE id ~ '^[0-9]+$'
            ) AS loan_rows,

            COUNT(*) FILTER (
                WHERE id !~ '^[0-9]+$'
                   OR id IS NULL
            ) AS metadata_rows,

            COUNT(DISTINCT id) FILTER (
                WHERE id ~ '^[0-9]+$'
            ) AS unique_loan_ids,

            COUNT(*) FILTER (
                WHERE id ~ '^[0-9]+$'
                  AND (
                      NULLIF(BTRIM(loan_amnt), '') IS NULL
                      OR NULLIF(BTRIM(loan_status), '') IS NULL
                      OR NULLIF(BTRIM(issue_d), '') IS NULL
                      OR NULLIF(BTRIM(grade), '') IS NULL
                  )
            ) AS incomplete_loans

        FROM raw.loans;
    """)

    columns_query = text("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'raw'
          AND table_name = 'loans';
    """)

    with engine.connect() as connection:
        results = connection.execute(query).mappings().one()
        column_count = connection.execute(
            columns_query
        ).scalar_one()

    checks = {
        "Expected source row count":
            results["total_rows"] == EXPECTED_TOTAL_ROWS,

        "Expected loan record count":
            results["loan_rows"] == EXPECTED_LOAN_ROWS,

        "Expected metadata record count":
            results["metadata_rows"] == EXPECTED_METADATA_ROWS,

        "All loan IDs are unique":
            results["unique_loan_ids"] == results["loan_rows"],

        "No missing critical loan fields":
            results["incomplete_loans"] == 0,

        "Expected column count":
            column_count == EXPECTED_COLUMNS,
    }

    print("\n=== RAW LOAN DATA QUALITY ===")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    print("\n=== RECORD SUMMARY ===")
    print(f"Total records: {results['total_rows']:,}")
    print(f"Loan records: {results['loan_rows']:,}")
    print(f"Metadata rows: {results['metadata_rows']:,}")
    print(f"Incomplete loans: {results['incomplete_loans']:,}")

    if not all(checks.values()):
        raise SystemExit("Data quality validation failed.")

    print("\nAll data quality checks passed.")


if __name__ == "__main__":
    check_raw_loans()
