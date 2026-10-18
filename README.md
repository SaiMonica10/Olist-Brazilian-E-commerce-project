# Olist Late-Delivery Risk

How much do late deliveries hurt review scores and repeat purchases, and can we
flag at-risk orders at the time of purchase so operations can intervene?

> **Status:** Phase 0 complete (database set up and raw data loaded). Analysis in progress.

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

<!-- TODO: after EDA, add the data-quality issues that shaped your decisions. -->

## Approach

<!-- TODO: fill in phase by phase as each one is done. -->

- **Phase 0:** Postgres in Docker, raw load, sanity checks
- **EDA:**
- **Measurement:**
- **Features and leakage control:**
- **Modeling:**
- **Serving:** FastAPI prediction API, Streamlit dashboard

## Findings

<!-- TODO: your findings, written after the analysis. Lead with numbers. -->

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

# 8. Open the notebooks in notebooks/ (in order) with the project's .venv kernel
```

## Project structure

```
data/raw/          Kaggle CSVs as downloaded (git-ignored)
data/processed/    Derived tables and feature sets (git-ignored)
notebooks/         Numbered exploration notebooks
sql/               Numbered SQL files
src/               Reusable Python: download, load, features, training
app/               FastAPI service and Streamlit dashboard
tests/             pytest tests
reports/figures/   Figures used in this README
```
