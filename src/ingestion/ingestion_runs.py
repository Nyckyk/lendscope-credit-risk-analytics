from sqlalchemy import text

from database import get_engine


def start_ingestion(source_file: str) -> int:
    engine = get_engine()

    query = text(
        """
        INSERT INTO meta.ingestion_runs (
            source_file,
            status
        )
        VALUES (
            :source_file,
            'RUNNING'
        )
        RETURNING run_id;
        """
    )

    with engine.begin() as connection:
        run_id = connection.execute(
            query,
            {"source_file": source_file},
        ).scalar_one()

    return int(run_id)


def mark_ingestion_success(
    run_id: int,
    rows_read: int,
    rows_loaded: int,
) -> None:
    engine = get_engine()

    query = text(
        """
        UPDATE meta.ingestion_runs
        SET
            completed_at = NOW(),
            status = 'SUCCESS',
            rows_read = :rows_read,
            rows_loaded = :rows_loaded,
            error_message = NULL
        WHERE run_id = :run_id
          AND status = 'RUNNING';
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "run_id": run_id,
                "rows_read": rows_read,
                "rows_loaded": rows_loaded,
            },
        )

        if result.rowcount != 1:
            raise RuntimeError(
                f"Could not mark run {run_id} as SUCCESS."
            )


def mark_ingestion_failed(
    run_id: int,
    error_message: str,
    rows_read: int = 0,
    rows_loaded: int = 0,
) -> None:
    engine = get_engine()

    query = text(
        """
        UPDATE meta.ingestion_runs
        SET
            completed_at = NOW(),
            status = 'FAILED',
            rows_read = :rows_read,
            rows_loaded = :rows_loaded,
            error_message = :error_message
        WHERE run_id = :run_id
          AND status = 'RUNNING';
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "run_id": run_id,
                "rows_read": rows_read,
                "rows_loaded": rows_loaded,
                "error_message": error_message,
            },
        )

        if result.rowcount != 1:
            raise RuntimeError(
                f"Could not mark run {run_id} as FAILED."
            )