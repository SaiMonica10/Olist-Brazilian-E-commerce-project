# Olist Late-Delivery Risk — Project Guide

## Goal

Portfolio project for a Junior Data Scientist role, built on the Olist Brazilian
E-commerce dataset (Kaggle `olistbr/brazilian-ecommerce`, 9 linked CSVs, ~100k
orders, 2016–2018).

Business question:

> How much do late deliveries hurt review scores and repeat purchases, and can we
> flag at-risk orders at the time of purchase so ops can intervene?

This breaks into two parts that must not be confused:

1. **Measurement** (descriptive/inferential): the effect of lateness on reviews
   and repeat purchase. Uses everything we know after the fact.
2. **Prediction**: a model that scores an order for late-delivery risk *at the
   moment it is placed*. Uses only what is known at purchase time.

## How to work with me

- I am learning and want to own every decision. Work in small steps, explain
  WHY for each choice, and pause for my confirmation before the next step.
- Do not write analysis conclusions or interpret results for me. Give me the
  query or the plot and the question it answers; the finding is mine to write.
- When there is a real trade-off, show me the options and your recommendation
  instead of silently picking one.

## Stack

- Python 3.11, dependencies managed with `uv` (`pyproject.toml` + `uv.lock`)
- PostgreSQL 16 in Docker (`docker-compose.yml`), accessed through SQLAlchemy
- pandas and SQL for analysis
- scikit-learn, LightGBM, SHAP for modeling
- Later phases: FastAPI (prediction API), Streamlit (dashboard), pytest

Add a dependency only in the phase that first uses it.

## Folder conventions

| Path               | Holds                                                        |
| ------------------ | ------------------------------------------------------------ |
| `data/raw/`        | Kaggle CSVs exactly as downloaded. Never edited. Git-ignored. |
| `data/processed/`  | Derived tables and feature sets. Git-ignored, reproducible.  |
| `notebooks/`       | Exploration only. No logic that other code depends on.       |
| `sql/`             | Numbered `.sql` files (`00_`, `01_`, ...).                   |
| `src/`             | Reusable, importable Python: loading, features, training.    |
| `models/`          | Trained model files from `src/train.py`. Git-ignored.        |
| `app/`             | FastAPI service and Streamlit dashboard (later phases).      |
| `tests/`           | pytest tests for `src/` and `app/`.                          |
| `reports/figures/` | Figures exported for the README and write-up.                |

If a notebook cell gets copied a second time, it moves into `src/` as a function.

## Rules

### 1. No leakage: features must exist at purchase time

Never use information unavailable at the moment of purchase as a model feature.

Why: the model's job is to flag an order when it is placed. A feature that only
exists later (actual delivery date, review score, carrier handoff date) makes
offline metrics look excellent and the deployed model useless.

The test for any feature: *could this value be computed at
`order_purchase_timestamp`, using only rows dated before it?* This applies to
aggregates too — a seller's historical late rate must be computed from that
seller's orders that were already completed before this order was placed.

Column availability (drafted in Phase 3, to be reviewed by me):

| Column | Known at purchase? | Notes |
| ------ | ------------------ | ----- |
| `orders.order_purchase_timestamp` | Yes | Defines "purchase time"; used for the time split, and month/day/hour features |
| `orders.order_estimated_delivery_date` | Yes | Shown to the customer at checkout → `promised_days` |
| `orders.order_approved_at` | No | Happens after purchase (boleto can take days) |
| `orders.order_delivered_carrier_date` | No | Carrier handoff, after purchase |
| `orders.order_delivered_customer_date` | No | This *is* the outcome (defines `is_late`) |
| `orders.order_status` | No | Final status, recorded later |
| `order_items.price`, `freight_value` | Yes | Shown in the basket |
| `order_items.product_id`, `seller_id` | Yes | Chosen at checkout |
| `order_items.shipping_limit_date` | No | Seller deadline, set ~6 days after approval |
| `order_payments.payment_type`, `payment_installments` | Yes | Chosen at checkout |
| `products.*` (category, weight, size) | Yes | Catalog attributes; assumed not to change over time |
| `customers.customer_state`, `customer_zip_code_prefix` | Yes | Delivery address |
| `sellers.seller_state`, `seller_zip_code_prefix` | Yes | Seller address |
| `geolocation.*` | Yes | Static zip → coordinates → `distance_km` |
| `order_reviews.*` | No | Written after (or during) delivery |
| Seller's past late rate | Only if time-aware | Count only that seller's orders delivered *before* this purchase |

### 2. `customer_unique_id` identifies a real customer

`customer_id` is generated per order; one person has a different `customer_id`
on every order. Anything about repeat purchase, customer history, or customer
counts must join through `customers` and use `customer_unique_id`.

Why: counting on `customer_id` makes every customer look like a one-time buyer,
which silently destroys the repeat-purchase half of the business question.

### 3. Notebook conventions

- Notebooks are numbered: `01_`, `02_`, ... in the order they should be read.
- Each starts with a markdown cell stating the single question it answers.

Why: a reviewer should be able to follow the project from filenames alone, and
a stated question keeps a notebook from sprawling.

## Data access

- Database credentials come from `.env` (never committed). `.env.example` is
  the committed template.
- `src/download_data.py` fetches the CSVs; `src/load_to_postgres.py` loads them.
  Both are safe to re-run.
- `sql/00_sanity_checks.sql` verifies a load.
- `sql/03_order_features.sql` builds the `order_features` materialized view;
  re-run it after every load. `src/features.py` loads it for modeling.
- `src/train.py` trains, evaluates and saves `models/late_risk_lgbm.joblib`;
  notebook 06 and `app/api.py` load that file. Re-run it after rebuilding
  `order_features`.
- `src/predict.py` holds the scoring code (build features, load model, score)
  shared by the API and the dashboard, so both always score the same way.
- `uv run pytest` runs the tests (no database needed, only the saved model).

## Phases

- **Phase 0** — scaffolding, database, data load, sanity checks
- **Phase 1** — EDA (notebooks 01–02)
- **Phase 2** — measurement (notebooks 03–04)
- **Phase 3** — purchase-time features (`sql/03_order_features.sql`, notebook 05)
- **Phase 4** — modeling (`src/train.py`, notebook 06): time split, logistic
  baseline vs LightGBM, SHAP. LightGBM kept, with the month feature.
- **Phase 5** — prediction API (`app/`): FastAPI, takes 19 features and
  computes `same_state` and `freight_ratio` itself, loads the model from `models/`
- **Phase 6** — Streamlit dashboard (`app/`): risk queue (replay a test-period
  day, slider for the share ops can check), score-an-order form, model page.
  Loads the model directly via `src/predict.py`; orders from `order_features`.
- Later phases (EDA, measurement, features, modeling, API, dashboard) are
  planned one at a time.
