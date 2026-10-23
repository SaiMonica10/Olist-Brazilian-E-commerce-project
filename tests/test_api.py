"""Tests for the prediction API (app/api.py).

Run from the project root:
    uv run pytest

Needs the saved model, so run `uv run python src/train.py` once first.
The database is not needed.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app
from predict import EXAMPLE_ORDER as RISKY_ORDER  # the riskiest test order, scored 44.0%

client = TestClient(app)


def predict(**changes):
    """POST the risky order with some fields changed (None removes a field)."""
    order = {**RISKY_ORDER, **changes}
    order = {k: v for k, v in order.items() if v is not None}
    return client.post("/predict", json=order)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.headers["location"] == "/docs"


def test_risky_order_matches_notebook():
    response = predict()
    assert response.status_code == 200
    assert response.json()["late_risk"] == pytest.approx(0.440, abs=0.001)


def test_safer_order_scores_lower():
    """Same order, but local delivery with a longer promise."""
    safer = predict(customer_state="SP", distance_km=15, promised_days=30)
    assert safer.json()["late_risk"] < predict().json()["late_risk"]


def test_optional_fields_can_be_missing():
    response = predict(distance_km=None, seller_prior_late_rate=None, main_category=None)
    assert response.status_code == 200
    assert 0 <= response.json()["late_risk"] <= 1


def test_zero_price_order_is_scored():
    """freight_ratio cannot be computed (division by zero), so it is left missing."""
    response = predict(total_price=0)
    assert response.status_code == 200
    assert 0 <= response.json()["late_risk"] <= 1


@pytest.mark.parametrize("field, value", [
    ("purchase_month_of_year", 13),
    ("purchase_hour", 24),
    ("seller_prior_late_rate", 1.5),
    ("customer_state", "XX"),
    ("seller_state", "sp"),
    ("main_category", "toys_typo"),
    ("main_payment_type", "cash"),
])
def test_bad_values_are_rejected(field, value):
    response = predict(**{field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_required_field_missing_is_rejected():
    assert predict(promised_days=None).status_code == 422


def test_real_state_unseen_for_sellers_is_accepted():
    """AC is a real state but no seller came from there in training."""
    assert predict(seller_state="AC").status_code == 200
