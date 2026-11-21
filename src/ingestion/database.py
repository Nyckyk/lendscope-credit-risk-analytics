import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


def get_database_url() -> str:
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    required = {
        "POSTGRES_HOST": host,
        "POSTGRES_PORT": port,
        "POSTGRES_DB": database,
        "POSTGRES_USER": user,
        "POSTGRES_PASSWORD": password,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return (
        f"postgresql+psycopg://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def get_engine():
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )


def test_connection() -> None:
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user;")
        )

        database, user = result.one()

    print("Database connection successful")
    print(f"Database: {database}")
    print(f"User: {user}")


if __name__ == "__main__":
    test_connection()