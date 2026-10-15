"""Database connection shared by scripts and notebooks.

Usage:
    from db import get_engine, query
    engine = get_engine()
    df = query("SELECT * FROM orders LIMIT 5")
"""

import os
from functools import cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, Engine, create_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@cache  # one engine (and its connection pool) per process
def get_engine() -> Engine:
    # Explicit path: notebooks run from notebooks/, so a plain load_dotenv()
    # would look for .env in the wrong folder.
    load_dotenv(PROJECT_ROOT / ".env")

    # URL.create escapes special characters in the password; an f-string would not.
    url = URL.create(
        "postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(url)


def query(sql: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql(sql, get_engine())
