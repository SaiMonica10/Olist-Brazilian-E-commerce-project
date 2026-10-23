"""Tests for src/predict.py, the scoring code shared by the API and dashboard."""

import math

from predict import EXAMPLE_ORDER as RISKY_ORDER
from predict import build_features, known_values, score
from train import FEATURES


def test_derived_features_match_the_sql_definitions():
    X = build_features([RISKY_ORDER, {**RISKY_ORDER, "customer_state": "SP"}])

    assert list(X.columns) == FEATURES
    assert X["same_state"].tolist() == [False, True]
    assert X["freight_ratio"].iloc[0] == 0.092  # 26.36 / 287.0, rounded to 3 decimals


def test_zero_price_leaves_freight_ratio_missing():
    X = build_features([{**RISKY_ORDER, "total_price": 0}])
    assert math.isnan(X["freight_ratio"].iloc[0])


def test_several_orders_scored_at_once():
    risks = score(build_features([RISKY_ORDER] * 3))
    assert risks.shape == (3,)
    assert ((risks >= 0) & (risks <= 1)).all()


def test_known_values_cover_all_states_for_sellers():
    known = known_values()
    assert len(known["customer_state"]) == 27
    assert known["seller_state"] == known["customer_state"]
