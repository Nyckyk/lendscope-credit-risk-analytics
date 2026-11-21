from sqlalchemy import text

from database import get_engine


def initialise_database() -> None:
    engine = get_engine()

    statements = [
        """
        CREATE SCHEMA IF NOT EXISTS raw;
        """,
        """
        CREATE SCHEMA IF NOT EXISTS meta;
        """,
        """
        CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
            run_id BIGSERIAL PRIMARY KEY,
            source_file TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            status TEXT NOT NULL,
            rows_read BIGINT NOT NULL DEFAULT 0,
            rows_loaded BIGINT NOT NULL DEFAULT 0,
            error_message TEXT
        );
        """,
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    print("Database schemas initialised successfully")
    print("Created/verified schema: raw")
    print("Created/verified schema: meta")
    print("Created/verified table: meta.ingestion_runs")


if __name__ == "__main__":
    initialise_database()