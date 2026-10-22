"""Load the purchase-time feature set built by sql/03_order_features.sql.

Usage:
    from features import load_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
    df = load_features()
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
"""

import pandas as pd

from db import query

TARGET = "is_late"

# Used for the time-based train/test split, never as a feature.
TIME_COLUMN = "order_purchase_timestamp"

NUMERIC_FEATURES = [
    "promised_days",
    "distance_km",
    "n_items",
    "n_products",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "max_installments",
    "purchase_month_of_year",
    "purchase_dow",
    "purchase_hour",
    "seller_prior_orders",
    # seller_prior_late is left out: it equals rate x orders, so it adds no
    # information and would split SHAP credit (notebook 05, section 6).
    "seller_prior_late_rate",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "same_state",
    "main_category",
    "main_payment_type",
]


def load_features() -> pd.DataFrame:
    """One row per delivered order (Jan 2017 - Aug 2018), sorted by purchase time."""
    df = query(f"SELECT * FROM order_features ORDER BY {TIME_COLUMN}, order_id")
    # Postgres numeric arrives as Python Decimal; models need floats.
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("category")
    df[TARGET] = df[TARGET].astype(int)
    return df
