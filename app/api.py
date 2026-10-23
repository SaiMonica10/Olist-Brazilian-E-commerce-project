"""Late-delivery risk API: scores one order at the moment it is placed.

Usage (from the project root):
    uv run uvicorn app.api:app --reload
    then open http://localhost:8000/docs to try it in the browser.

Decisions:
    * The caller sends 19 of the 21 purchase-time features ready-made (the
      columns of the order_features view). The API computes the other two,
      same_state and freight_ratio, from the fields they are derived from, so
      they can never contradict them. Computing all features from raw order
      details is a later improvement.
    * Features that can be missing in the data (distance, weight, volume, a new
      seller's late rate, category) are optional; LightGBM handles missing
      values itself.
    * Text fields only accept known values, so a typo gets a clear error instead
      of a quiet guess. States: the 27 Brazilian states. A real state the
      model never saw for sellers is treated like a missing value.
      Category and payment type: the values the model was trained on.
    * late_risk is best used to rank orders (higher = check first). As a
      probability it is only approximate: on the test months the model
      predicted 5.7% late on average, while 4.4% actually were.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

# app/ is not a package of src/, so add src/ to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import EXAMPLE_ORDER, build_features, known_values, score  # noqa: E402

KNOWN = known_values()  # also loads the model at startup, not on the first request

app = FastAPI(
    title="Olist late-delivery risk",
    description="Scores an order's risk of arriving after the promised date, "
                "using only what is known at purchase time.",
)


class Order(BaseModel):
    """The purchase-time features of one order (see sql/03_order_features.sql),
    except same_state and freight_ratio, which build_features() computes."""

    promised_days: float = Field(ge=0, description="Days between purchase and the promised delivery date")
    distance_km: float | None = Field(None, ge=0, description="Seller-to-customer distance")
    n_items: int = Field(ge=1)
    n_products: int = Field(ge=1)
    n_sellers: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    total_weight_g: float | None = Field(None, ge=0)
    total_volume_cm3: float | None = Field(None, ge=0)
    max_installments: int = Field(ge=0)
    purchase_month_of_year: int = Field(ge=1, le=12)
    purchase_dow: int = Field(ge=1, le=7, description="1 = Monday")
    purchase_hour: int = Field(ge=0, le=23)
    seller_prior_orders: int = Field(ge=0, description="Seller's orders delivered before this purchase")
    seller_prior_late_rate: float | None = Field(None, ge=0, le=1, description="Empty for a new seller")
    customer_state: str = Field(description="e.g. SP")
    seller_state: str
    main_category: str | None = None
    main_payment_type: str = Field(description="credit_card, boleto, voucher or debit_card")

    @field_validator("customer_state", "seller_state", "main_category", "main_payment_type")
    @classmethod
    def must_be_known(cls, value, info):
        if value is not None and value not in KNOWN[info.field_name]:
            raise ValueError(f"unknown value {value!r}; see GET /categories for the allowed values")
        return value

    # Shown as the ready-filled example on the /docs page: the riskiest test order.
    model_config = {"json_schema_extra": {"examples": [EXAMPLE_ORDER]}}


class Prediction(BaseModel):
    late_risk: float = Field(description="Late-delivery risk score from 0 to 1; higher = check first. "
                                         "Roughly a probability, but best used for ranking.")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The bare address has no page of its own; send visitors to the test page."""
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/categories")
def categories() -> dict[str, list[str]]:
    """Allowed values for each text field."""
    return {name: KNOWN[name] for name in
            ["customer_state", "seller_state", "main_category", "main_payment_type"]}


@app.post("/predict")
def predict(order: Order) -> Prediction:
    risk = score(build_features([order.model_dump()]))[0]
    return Prediction(late_risk=round(float(risk), 4))
