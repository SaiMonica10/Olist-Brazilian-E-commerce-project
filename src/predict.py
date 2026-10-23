"""Score orders with the saved late-delivery model.

Shared by the API (app/api.py) and the dashboard, so both always score an
order the same way.

Usage:
    from predict import build_features, load_model, score
    X = build_features([order_dict])   # 19 fields, as the API receives them
    risk = score(X)                     # array of risks, one per order
"""

from functools import cache

import joblib
import numpy as np
import pandas as pd

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train import FEATURES, MODEL_PATH


# The riskiest order in the test set (notebook 06, section 4), as the API
# receives it. Used as the example in the API docs, the tests and the dashboard form.
EXAMPLE_ORDER = {
    "promised_days": 19, "distance_km": 1968.3, "n_items": 1, "n_products": 1,
    "n_sellers": 1, "total_price": 287.0, "total_freight": 26.36,
    "total_weight_g": 650, "total_volume_cm3": 1188, "max_installments": 3,
    "purchase_month_of_year": 8, "purchase_dow": 4, "purchase_hour": 18,
    "seller_prior_orders": 21, "seller_prior_late_rate": 0.048,
    "customer_state": "AL", "seller_state": "SP",
    "main_category": "health_beauty", "main_payment_type": "credit_card",
}


@cache  # load the file once per process
def load_model():
    return joblib.load(MODEL_PATH)


@cache
def known_values() -> dict[str, list]:
    """Allowed values for each text field: the categories saved with the model.
    Customers cover all 27 states, so that list is used for sellers too."""
    known = dict(zip(CATEGORICAL_FEATURES, load_model().booster_.pandas_categorical))
    known["seller_state"] = known["customer_state"]
    return known


def build_features(orders: list[dict]) -> pd.DataFrame:
    """Turn raw order fields into the model's feature table.

    Computes same_state and freight_ratio exactly as sql/03_order_features.sql
    does, and sets the same column types as load_features().
    """
    X = pd.DataFrame(orders)
    X["same_state"] = X["customer_state"] == X["seller_state"]
    X["freight_ratio"] = (X["total_freight"] / X["total_price"].where(X["total_price"] > 0)).round(3)
    X = X.reindex(columns=FEATURES)  # a missing optional field becomes an empty column
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    # LightGBM matches categories by name, using the list saved with the model.
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype("category")
    return X


def score(X: pd.DataFrame) -> np.ndarray:
    """Late-delivery risk (0 to 1) for each row of a feature table."""
    return load_model().predict_proba(X[FEATURES])[:, 1]


@cache
def _explainer():
    import shap  # imported here: slow to import, and the API does not need it

    return shap.TreeExplainer(load_model())


def explain(X: pd.DataFrame):
    """SHAP values: each feature's push on each order's risk (in log-odds,
    positive = toward late). Returns a shap.Explanation, one row per order."""
    return _explainer()(X[FEATURES])
