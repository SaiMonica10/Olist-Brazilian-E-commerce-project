"""Load the 9 raw Olist CSVs from data/raw/ into PostgreSQL.

Safe to re-run: every table is dropped and recreated, and the whole load runs
in one transaction, so a failure leaves the database as it was.

Tables mirror the CSVs exactly (same column names, no cleaning). Cleaning and
renaming happen later, downstream of the raw tables.

Usage:
    docker compose up -d
    uv run python src/load_to_postgres.py
"""

from pathlib import Path

from sqlalchemy import Connection, text

from db import get_engine

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# table -> (CSV file, column definitions).
# Columns must be in the same order as the CSV header; COPY ... HEADER MATCH
# below fails loudly if they are not.

TABLES = {
    "customers": (
        "olist_customers_dataset.csv",
        
        """
        customer_id              TEXT PRIMARY KEY,
        customer_unique_id       TEXT NOT NULL,
        customer_zip_code_prefix TEXT,
        customer_city            TEXT,
        customer_state           CHAR(2)
        """,

    ),
    "geolocation": (
        # No primary key: a zip prefix appears many times with different coordinates.
        "olist_geolocation_dataset.csv",
        """
        geolocation_zip_code_prefix TEXT,
        geolocation_lat             DOUBLE PRECISION,
        geolocation_lng             DOUBLE PRECISION,
        geolocation_city            TEXT,
        geolocation_state           CHAR(2)
        """,
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        """
        order_id            TEXT,
        order_item_id       SMALLINT,
        product_id          TEXT,
        seller_id           TEXT,
        shipping_limit_date TIMESTAMP,
        price               NUMERIC(10, 2),
        freight_value       NUMERIC(10, 2),
        PRIMARY KEY (order_id, order_item_id)
        """,
    ),
    "order_payments": (
        "olist_order_payments_dataset.csv",
        """
        order_id             TEXT,
        payment_sequential   SMALLINT,
        payment_type         TEXT,
        payment_installments SMALLINT,
        payment_value        NUMERIC(10, 2),
        PRIMARY KEY (order_id, payment_sequential)
        """,
    ),
    "order_reviews": (
        # review_id alone is not unique (814 repeats), so the key is composite.
        "olist_order_reviews_dataset.csv",
        """
        review_id               TEXT,
        order_id                TEXT,
        review_score            SMALLINT,
        review_comment_title    TEXT,
        review_comment_message  TEXT,
        review_creation_date    TIMESTAMP,
        review_answer_timestamp TIMESTAMP,
        PRIMARY KEY (review_id, order_id)
        """,
    ),
    "orders": (
        "olist_orders_dataset.csv",
        """
        order_id                      TEXT PRIMARY KEY,
        customer_id                   TEXT NOT NULL,
        order_status                  TEXT,
        order_purchase_timestamp      TIMESTAMP,
        order_approved_at             TIMESTAMP,
        order_delivered_carrier_date  TIMESTAMP,
        order_delivered_customer_date TIMESTAMP,
        order_estimated_delivery_date TIMESTAMP
        """,
    ),
    "products": (
        # "lenght" is misspelled in the source; kept so the table matches the CSV.
        "olist_products_dataset.csv",
        """
        product_id                 TEXT PRIMARY KEY,
        product_category_name      TEXT,
        product_name_lenght        INTEGER,
        product_description_lenght INTEGER,
        product_photos_qty         INTEGER,
        product_weight_g           INTEGER,
        product_length_cm          INTEGER,
        product_height_cm          INTEGER,
        product_width_cm           INTEGER
        """,
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        """
        seller_id              TEXT PRIMARY KEY,
        seller_zip_code_prefix TEXT,
        seller_city            TEXT,
        seller_state           CHAR(2)
        """,
    ),
    "product_category_translation": (
        "product_category_name_translation.csv",
        """
        product_category_name         TEXT PRIMARY KEY,
        product_category_name_english TEXT
        """,
    ),
      
}

def load_table(conn: Connection, table: str, csv_name: str, columns: str) -> int:
    # CASCADE also drops views built on this table (e.g. order_delivery);
    # re-create them afterwards by running the sql/ files in order.
    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    conn.execute(text(f"CREATE TABLE {table} ({columns})"))
    
    # COPY streams the file straight into Postgres, which parses and type-checks
    # every value. utf-8-sig strips the byte-order mark at the start of the
    # translation CSV, which would otherwise break HEADER MATCH.
    cursor = conn.connection.cursor()
    with cursor.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER MATCH)") as copy:
        with open(RAW_DIR / csv_name, encoding="utf-8-sig") as f:
            while chunk := f.read(1_000_000):
                copy.write(chunk)

    return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:  # commits at the end, rolls back on any error
        for table, (csv_name, columns) in TABLES.items():
            rows = load_table(conn, table, csv_name, columns)
            print(f"  {table:<30} {rows:>10,} rows")


if __name__ == "__main__":
    main()