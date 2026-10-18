-- 02_order_outcomes.sql
-- One row per delivered order in the analysis period, with the outcomes that
-- Phase 2 measures: review score and repeat purchase. Builds on order_delivery
-- (sql/01_order_delivery.sql), so run that first.
--
-- Run after loading (safe to re-run):
--   docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/02_order_outcomes.sql
--
-- Decisions:
--   * Period: purchases Jan 2017 - Aug 2018 (notebook 01: 2016 is too sparse,
--     Sep-Oct 2018 has no delivered orders).
--   * Review: if an order has several reviews, the latest answered one counts
--     (the customer's final opinion).
--   * Repeat purchase: the same customer_unique_id places another order at
--     least 1 day later. Orders minutes apart are one shopping session, not a
--     return visit. Any order status counts: the customer chose to buy again.
--   * Window: 180 days after purchase. repeat_within_180d is NULL when the
--     window runs past the end of the data (2018-09-01), so those orders are
--     not wrongly counted as "did not return".
--   * customer_order_number counts all of a customer's orders, any status,
--     from the first order in the data.

CREATE OR REPLACE VIEW order_outcomes AS
WITH customer_orders AS (
    SELECT order_id,
           customer_unique_id,
           order_purchase_timestamp,
           ROW_NUMBER() OVER (PARTITION BY customer_unique_id
                              ORDER BY order_purchase_timestamp, order_id) AS customer_order_number
    FROM order_delivery
),
next_purchase AS (
    SELECT a.order_id,
           MIN(b.order_purchase_timestamp) AS next_purchase_timestamp
    FROM customer_orders a
    JOIN customer_orders b
      ON b.customer_unique_id = a.customer_unique_id
     AND b.order_purchase_timestamp >= a.order_purchase_timestamp + INTERVAL '1 day'
    GROUP BY a.order_id
),
latest_review AS (
    SELECT DISTINCT ON (order_id)
           order_id,
           review_score,
           review_creation_date
    FROM order_reviews
    ORDER BY order_id, review_answer_timestamp DESC, review_creation_date DESC
)
SELECT
    d.order_id,
    d.customer_unique_id,
    d.customer_state,
    d.order_purchase_timestamp,
    date_trunc('month', d.order_purchase_timestamp)::date          AS purchase_month,
    d.order_delivered_customer_date,
    d.order_estimated_delivery_date,
    d.days_late,
    d.is_late,

    r.review_score,                                                -- NULL: no review
    r.review_creation_date,
    r.review_creation_date::date
        < d.order_delivered_customer_date::date                    AS review_before_delivery,

    c.customer_order_number,
    c.customer_order_number = 1                                    AS is_first_order,
    n.next_purchase_timestamp,
    CASE WHEN d.order_purchase_timestamp + INTERVAL '180 days' <= TIMESTAMP '2018-09-01'
         THEN COALESCE(n.next_purchase_timestamp
                       <= d.order_purchase_timestamp + INTERVAL '180 days', FALSE)
    END                                                            AS repeat_within_180d
FROM order_delivery d
JOIN customer_orders c        USING (order_id)
LEFT JOIN next_purchase n     USING (order_id)
LEFT JOIN latest_review r     USING (order_id)
WHERE d.is_delivered
  AND d.order_purchase_timestamp >= TIMESTAMP '2017-01-01'
  AND d.order_purchase_timestamp <  TIMESTAMP '2018-09-01';
