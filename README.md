# Olist Late-Delivery Risk

How much do late deliveries hurt review scores and repeat purchases, and can we
flag at-risk orders at the time of purchase so operations can intervene?

> **Status:** Phases 0–5 complete (data load, EDA, measurement, features,
> model, prediction API). Phase 6 (Streamlit dashboard) next.

## Problem

<!-- TODO: 2-4 sentences in your own words. Who is the stakeholder (ops team)?
     What decision would this project support? Why does it matter to Olist? -->

The project has two separate parts:

1. **Measurement:** how late delivery relates to review scores and repeat
   purchase, using everything known after the order is complete.
2. **Prediction:** a model that scores an order's late-delivery risk at the
   moment it is placed, using only information available at purchase time.

## Data

[Olist Brazilian E-commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
about 100k orders from 2016 to 2018, made public by Olist, a Brazilian
marketplace platform. It comes as 9 CSVs, loaded into PostgreSQL as these tables:

| Table | Rows | One row per |
| ----- | ---: | ----------- |
| `orders` | 99,441 | order |
| `order_items` | 112,650 | item within an order |
| `order_payments` | 103,886 | payment within an order |
| `order_reviews` | 99,224 | review of an order |
| `customers` | 99,441 | order's customer record (see note) |
| `products` | 32,951 | product |
| `sellers` | 3,095 | seller |
| `geolocation` | 1,000,163 | zip prefix / coordinate pair (several per zip) |
| `product_category_translation` | 71 | category (Portuguese → English) |

Notes:

- `customer_id` is a new value for every order. The real customer is
  `customer_unique_id`, and repeat-purchase analysis depends on it.
- Raw tables mirror the CSVs exactly; data-quality checks are in
  [`sql/00_sanity_checks.sql`](sql/00_sanity_checks.sql).

Data-quality issues that shaped the analysis
([notebook 01](notebooks/01_data_overview.ipynb)):

- **Scope:** lateness is measured on the 96,470 orders that are `delivered`
  and have a delivery date (97% of orders).
- **Period:** purchases from Jan 2017 to Aug 2018. 2016 is too sparse (one
  month has no orders), and Sep–Oct 2018 has no delivered orders.
- **Out-of-order middle dates:** 1,373 delivered orders have approval or
  carrier dates out of order. They are kept, because purchase and delivery
  dates are always valid, so their late label is correct.
- **Limitation:** most of the 1,107 `shipped` orders are old and were likely
  never delivered. They are excluded, so measured lateness is slightly
  optimistic.

## Approach

<!-- TODO: fill in phase by phase as each one is done. -->

- **Phase 0:** Postgres in Docker, raw load, sanity checks
- **EDA:** which orders to analyse and how lateness varies over time and by
  state ([01](notebooks/01_data_overview.ipynb),
  [02](notebooks/02_delivery_lateness.ipynb)). "Late" is defined once, in
  [`sql/01_order_delivery.sql`](sql/01_order_delivery.sql): delivered on a
  later date than the estimated date.
- **Measurement:** late vs. on-time orders compared on review score and
  180-day repeat purchase, with 95% confidence intervals, a regression that
  adjusts for customer state and purchase month, and a robustness check
  ([03](notebooks/03_lateness_vs_reviews.ipynb),
  [04](notebooks/04_lateness_vs_repeat.ipynb)). Outcomes are built in
  [`sql/02_order_outcomes.sql`](sql/02_order_outcomes.sql).
- **Features and leakage control:** one row per order with features known at
  purchase time: promised delivery days, seller–customer distance, states,
  basket contents, payment type, purchase time, and the seller's past late
  rate ([`sql/03_order_features.sql`](sql/03_order_features.sql),
  [notebook 05](notebooks/05_features.ipynb)). The seller's history counts
  only orders already delivered *before* the purchase, so the model never
  sees the future; a brute-force recount on 500 random orders matched
  exactly. Post-purchase columns (approval, carrier and delivery dates,
  shipping deadline, reviews) are excluded.
- **Modeling:** time-based split (train on purchases before Mar 2018,
  validate on Mar–Apr 2018, test once on May–Aug 2018). A random-score
  baseline, logistic regression and LightGBM are compared on average
  precision (only ~7% of orders are late, so accuracy is not used) and on
  precision/recall when the riskiest 10% of orders are flagged
  ([`src/train.py`](src/train.py)). LightGBM is explained with SHAP
  ([notebook 06](notebooks/06_model_explain.ipynb)).
- **Serving:** FastAPI prediction API ([`app/api.py`](app/api.py)) that takes
  an order's purchase-time features and returns its risk score, with input
  checks and pytest tests. Features that can be derived (`same_state`,
  `freight_ratio`) are computed by the API, so they can't contradict the input. Streamlit dashboard next.

## Findings

**How often orders are late** ([02](notebooks/02_delivery_lateness.ipynb))

- 6.8% of delivered orders arrive late. A typical order arrives 12 days
  before its estimate, so Olist's estimates are generous.
- Lateness comes in bursts: Nov 2017, Feb 2018 and Mar 2018 hold 48% of all
  late orders.
- The late rate varies by customer state, from 2.8% (AM) to 21.4% (AL).

**Lateness and review scores** ([03](notebooks/03_lateness_vs_reviews.ipynb))

- Late orders average **2.27 stars vs 4.29** for on-time orders, a gap of
  **−2.02 stars**. Comparing orders from the same state and month, the gap is
  still −1.96 (95% CI −2.00 to −1.92).
- 53.8% of late orders get 1 star, vs 6.6% of on-time orders.
- Most of the gap comes from reviews written while customers were still
  waiting. Counting only reviews written after arrival, the gap is
  **−0.77 stars** (95% CI −0.84 to −0.69).

**Lateness and repeat purchase** ([04](notebooks/04_lateness_vs_repeat.ipynb))

- Only 2.0% of customers buy again within 180 days.
- Customers whose first order was late came back at 1.58% vs 2.01%
  (−0.43 percentage points). Within the same state and month the gap is
  −0.26 points (95% CI −0.69 to +0.16), which includes zero.
- **Inconclusive:** repeat purchase is too rare in this data to detect a
  small effect. The case for a late-delivery model therefore rests on
  review scores.

All of these are associations, not proof of cause: product, seller and price
are not adjusted for.

## Model

<!-- TODO: target definition, features, validation scheme (time-based split),
     metrics vs. a baseline, SHAP drivers, how ops would use the score. -->

## How to run

Requirements: [uv](https://docs.astral.sh/uv/), Docker Desktop, and a
[Kaggle API token](https://www.kaggle.com/docs/api).

```bash
# 1. Install dependencies
uv sync

# 2. Configure credentials (then edit .env and set a password)
cp .env.example .env

# 3. Start PostgreSQL 16
docker compose up -d

# 4. Download the dataset into data/raw/
uv run python src/download_data.py

# 5. Load the CSVs into Postgres
uv run python src/load_to_postgres.py

# 6. Verify the load
docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/00_sanity_checks.sql

# 7. Create the analysis views (re-run after every load)
docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/01_order_delivery.sql
docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/02_order_outcomes.sql
docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/03_order_features.sql

# 8. Open the notebooks in notebooks/ (in order) with the project's .venv kernel

# 9. Train and evaluate the models; saves models/late_risk_lgbm.joblib
#    (notebook 06 and the API load this file)
uv run python src/train.py

# 10. Run the tests (no database needed, only the saved model)
uv run pytest

# 11. Start the prediction API, then open http://localhost:8000/docs
uv run uvicorn app.api:app --reload
```

On macOS, LightGBM needs OpenMP: `brew install libomp`.

## Project structure

```
data/raw/          Kaggle CSVs as downloaded (git-ignored)
data/processed/    Derived tables and feature sets (git-ignored)
notebooks/         Numbered exploration notebooks
sql/               Numbered SQL files
src/               Reusable Python: download, load, features, training
models/            Trained model saved by src/train.py (git-ignored)
app/               FastAPI service and Streamlit dashboard
tests/             pytest tests
reports/figures/   Figures used in this README
```
