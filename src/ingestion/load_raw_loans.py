from pathlib import Path

from psycopg import sql
from sqlalchemy import text

from database import get_engine
from ingestion_runs import (
    mark_ingestion_failed,
    mark_ingestion_success,
    start_ingestion,
)
from source_contract import (
    SOURCE_PATH,
    validate_source_schema,
)


COPY_BUFFER_SIZE = 1024 * 1024


def create_staging_table(
    table_name: str,
    columns: list[str],
) -> None:
    engine = get_engine()

    column_definitions = sql.SQL(", ").join(
        sql.SQL("{} TEXT").format(
            sql.Identifier(column)
        )
        for column in columns
    )

    create_statement = sql.SQL(
        "CREATE TABLE {}.{} ({})"
    ).format(
        sql.Identifier("raw"),
        sql.Identifier(table_name),
        column_definitions,
    )

    with engine.raw_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_statement)

        connection.commit()


def copy_csv_to_staging(
    table_name: str,
    columns: list[str],
    source_path: Path,
) -> None:
    engine = get_engine()

    column_list = sql.SQL(", ").join(
        sql.Identifier(column)
        for column in columns
    )

    copy_statement = sql.SQL(
        """
        COPY {}.{} ({})
        FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE
        )
        """
    ).format(
        sql.Identifier("raw"),
        sql.Identifier(table_name),
        column_list,
    )

    with engine.raw_connection() as connection:
        with connection.cursor() as cursor:
            with cursor.copy(copy_statement) as copy:
                with source_path.open("rb") as source_file:
                    while True:
                        block = source_file.read(
                            COPY_BUFFER_SIZE
                        )

                        if not block:
                            break

                        copy.write(block)

        connection.commit()


def count_rows(table_name: str) -> int:
    engine = get_engine()

    query = text(
        f"""
        SELECT COUNT(*)
        FROM raw."{table_name}";
        """
    )

    with engine.connect() as connection:
        count = connection.execute(
            query
        ).scalar_one()

    return int(count)


def promote_staging_table(
    staging_table: str,
) -> None:
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DROP TABLE IF EXISTS raw.loans_previous;
                """
            )
        )

        existing_loans = connection.execute(
            text(
                """
                SELECT to_regclass('raw.loans');
                """
            )
        ).scalar_one()

        if existing_loans is not None:
            connection.execute(
                text(
                    """
                    ALTER TABLE raw.loans
                    RENAME TO loans_previous;
                    """
                )
            )

        connection.execute(
            text(
                f"""
                ALTER TABLE raw."{staging_table}"
                RENAME TO loans;
                """
            )
        )

        connection.execute(
            text(
                """
                DROP TABLE IF EXISTS raw.loans_previous;
                """
            )
        )


def drop_staging_table(
    table_name: str,
) -> None:
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                DROP TABLE IF EXISTS raw."{table_name}";
                """
            )
        )


def load_raw_loans() -> None:
    source_file = SOURCE_PATH.name

    print("Validating source schema...")

    columns = validate_source_schema()

    run_id = start_ingestion(
        source_file=source_file
    )

    staging_table = (
        f"loans_stage_{run_id}"
    )

    print()
    print(f"Ingestion run: {run_id}")
    print(
        f"Staging table: raw.{staging_table}"
    )

    try:
        print()
        print("Creating staging table...")

        create_staging_table(
            staging_table,
            columns,
        )

        print("Staging table created.")

        print()
        print("Bulk loading CSV into PostgreSQL...")

        copy_csv_to_staging(
            staging_table,
            columns,
            SOURCE_PATH,
        )

        print("COPY completed.")

        print()
        print("Counting loaded rows...")

        rows_loaded = count_rows(
            staging_table
        )

        print(
            f"Rows loaded: {rows_loaded:,}"
        )

        if rows_loaded == 0:
            raise RuntimeError(
                "Staging table contains no rows."
            )

        print()
        print(
            "Promoting staging table "
            "to raw.loans..."
        )

        promote_staging_table(
            staging_table
        )

        mark_ingestion_success(
            run_id=run_id,
            rows_read=rows_loaded,
            rows_loaded=rows_loaded,
        )

        print()
        print("Ingestion completed successfully.")
        print(
            f"Run ID: {run_id}"
        )
        print(
            f"Rows: {rows_loaded:,}"
        )
        print(
            "Final table: raw.loans"
        )

    except Exception as exc:
        try:
            drop_staging_table(
                staging_table
            )
        except Exception:
            pass

        mark_ingestion_failed(
            run_id=run_id,
            error_message=str(exc),
        )

        print()
        print("Ingestion FAILED.")
        print(
            f"Run ID: {run_id}"
        )
        print(
            f"Error: {exc}"
        )

        raise


if __name__ == "__main__":
    load_raw_loans()