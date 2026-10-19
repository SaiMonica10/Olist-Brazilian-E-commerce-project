-- 03_order_features.sql
-- One row per delivered order in the analysis period, with model features that
-- are known at the moment of purchase, plus the is_late label. Builds on
-- order_delivery (sql/01_order_delivery.sql), so run that first.
--
-- Run after loading (safe to re-run):
--   docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/03_order_features.sql
--
-- Decisions:
--   * Rows: delivered orders bought Jan 2017 - Aug 2018 (same scope as
--     order_outcomes). order_purchase_timestamp is kept for the time-based
--     train/test split; it is not a feature itself.
--   * Leakage rule (CLAUDE.md rule 1): every feature must be computable at
--     order_purchase_timestamp. Excluded for that reason: approval, carrier
--     and delivery dates, shipping_limit_date (set after approval), reviews.
--   * Main seller / main category: from the order's highest-priced item.
--     1.3% of orders have several sellers; n_sellers flags them.
--   * Distance: straight-line km between the customer's and the main seller's
--     zip prefix, using the average coordinate per zip. Coordinates outside
--     Brazil's bounding box are dropped as errors. NULL when a zip is missing
--     from geolocation.
--   * Seller history: counts only the main seller's orders that were already
--     DELIVERED before this order was placed. An earlier order that was still
--     in transit at purchase time has no known outcome yet, so it is not
--     counted. History uses all delivered orders, including 2016.
--   * Product size and geolocation are static tables (a snapshot), assumed
--     not to change over time.
--   * Materialized (computed once and stored) instead of a plain view: the
--     model reads it many times. Re-running this file rebuilds it.
--   * enable_nestloop is turned off only while building: Postgres badly
--     underestimates the CTE row counts and picks a plan that takes 5+ minutes
--     instead of under a second.

DROP MATERIALIZED VIEW IF EXISTS order_features;

BEGIN;
SET LOCAL enable_nestloop = off;

CREATE MATERIALIZED VIEW order_features AS
WITH zip_coords AS (
    SELECT geolocation_zip_code_prefix AS zip,
           AVG(geolocation_lat)        AS lat,
           AVG(geolocation_lng)        AS lng
    FROM geolocation
    WHERE geolocation_lat BETWEEN -34 AND 6
      AND geolocation_lng BETWEEN -74 AND -34
    GROUP BY geolocation_zip_code_prefix
),
item_lines AS (
    SELECT i.order_id,
           i.order_item_id,
           i.product_id,
           i.seller_id,
           i.price,
           i.freight_value,
           p.product_weight_g,
           p.product_length_cm * p.product_height_cm * p.product_width_cm AS volume_cm3,
           COALESCE(t.product_category_name_english, p.product_category_name) AS category
    FROM order_items i
    LEFT JOIN products p                     USING (product_id)
    LEFT JOIN product_category_translation t USING (product_category_name)
),
items_agg AS (
    SELECT order_id,
           COUNT(*)                   AS n_items,
           COUNT(DISTINCT product_id) AS n_products,
           COUNT(DISTINCT seller_id)  AS n_sellers,
           SUM(price)                 AS total_price,
           SUM(freight_value)         AS total_freight,
           SUM(product_weight_g)      AS total_weight_g,
           SUM(volume_cm3)            AS total_volume_cm3
    FROM item_lines
    GROUP BY order_id
),
main_item AS (
    SELECT DISTINCT ON (order_id)
           order_id,
           seller_id AS main_seller_id,
           category  AS main_category
    FROM item_lines
    ORDER BY order_id, price DESC, order_item_id
),
payments_agg AS (
    SELECT order_id,
           MAX(payment_installments) AS max_installments
    FROM order_payments
    GROUP BY order_id
),
main_payment AS (
    SELECT DISTINCT ON (order_id)
           order_id,
           payment_type AS main_payment_type
    FROM order_payments
    ORDER BY order_id, payment_value DESC, payment_sequential
),
-- Seller history without a slow per-order lookup: put each seller's
-- deliveries and purchases on one timeline, then keep a running count of
-- deliveries. At a purchase row, the running count = deliveries before it.
seller_timeline AS (
    SELECT DISTINCT i.seller_id,
           d.order_delivered_customer_date AS event_time,
           1                               AS is_delivery,  -- sorts after a purchase at the same instant
           d.order_id,
           d.is_late
    FROM order_items i
    JOIN order_delivery d USING (order_id)
    WHERE d.is_delivered
    UNION ALL
    SELECT m.main_seller_id,
           o.order_purchase_timestamp,
           0,
           o.order_id,
           NULL
    FROM main_item m
    JOIN orders o USING (order_id)
),
seller_history AS (
    SELECT order_id, prior_orders, prior_late
    FROM (
        SELECT order_id,
               is_delivery,
               SUM(is_delivery) OVER w                    AS prior_orders,
               SUM((is_late IS TRUE)::int) OVER w         AS prior_late
        FROM seller_timeline
        WINDOW w AS (PARTITION BY seller_id
                     ORDER BY event_time, is_delivery
                     ROWS UNBOUNDED PRECEDING)
    ) t
    WHERE is_delivery = 0
)
SELECT
    d.order_id,
    d.order_purchase_timestamp,                                    -- for the time split only
    d.is_late,                                                     -- label

    -- Promise
    d.order_estimated_delivery_date::date
        - d.order_purchase_timestamp::date                         AS promised_days,

    -- Geography
    d.customer_state,
    s.seller_state,
    d.customer_state = s.seller_state                              AS same_state,
    ROUND((2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(sz.lat - cz.lat) / 2), 2)
        + COS(RADIANS(cz.lat)) * COS(RADIANS(sz.lat))
          * POWER(SIN(RADIANS(sz.lng - cz.lng) / 2), 2))))::numeric, 1)
                                                                   AS distance_km,

    -- Order contents
    ia.n_items,
    ia.n_products,
    ia.n_sellers,
    ia.total_price,
    ia.total_freight,
    ROUND(ia.total_freight / NULLIF(ia.total_price, 0), 3)         AS freight_ratio,
    ia.total_weight_g,
    ia.total_volume_cm3,
    mi.main_category,

    -- Payment (chosen at checkout)
    mp.main_payment_type,
    pa.max_installments,

    -- Time of purchase
    EXTRACT(MONTH FROM d.order_purchase_timestamp)::int            AS purchase_month_of_year,
    EXTRACT(ISODOW FROM d.order_purchase_timestamp)::int           AS purchase_dow,   -- 1 = Monday
    EXTRACT(HOUR FROM d.order_purchase_timestamp)::int             AS purchase_hour,

    -- Main seller's track record before this purchase
    sh.prior_orders                                                AS seller_prior_orders,
    sh.prior_late                                                  AS seller_prior_late,
    ROUND(sh.prior_late::numeric / NULLIF(sh.prior_orders, 0), 3)  AS seller_prior_late_rate  -- NULL: no history
FROM order_delivery d
JOIN customers c          USING (customer_id)
JOIN items_agg ia         USING (order_id)
JOIN main_item mi         USING (order_id)
JOIN sellers s            ON s.seller_id = mi.main_seller_id
LEFT JOIN payments_agg pa USING (order_id)
LEFT JOIN main_payment mp USING (order_id)
LEFT JOIN seller_history sh USING (order_id)
LEFT JOIN zip_coords cz   ON cz.zip = c.customer_zip_code_prefix
LEFT JOIN zip_coords sz   ON sz.zip = s.seller_zip_code_prefix
WHERE d.is_delivered
  AND d.order_purchase_timestamp >= TIMESTAMP '2017-01-01'
  AND d.order_purchase_timestamp <  TIMESTAMP '2018-09-01';

CREATE UNIQUE INDEX order_features_order_id ON order_features (order_id);

COMMIT;
