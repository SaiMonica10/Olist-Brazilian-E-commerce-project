-- 01_order_delivery.sql
-- One row per order with its delivery outcome. This is the single definition
-- of "late" that every notebook and the model use.
--
-- Run after loading (safe to re-run):
--   docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/01_order_delivery.sql
--
-- Decisions:
--   * Late = delivered on a later DATE than the estimated date. The estimate is
--     always midnight, so comparing timestamps would mark same-day deliveries late.
--   * Lateness is only defined for delivered orders (status 'delivered' with a
--     delivery date). Other orders stay in the view with NULL lateness, so
--     nothing is silently dropped; filter on is_delivered to analyse lateness.
--   * customer_unique_id is included because it, not customer_id, identifies
--     a real person (needed for repeat-purchase analysis).

CREATE OR REPLACE VIEW order_delivery AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,

    (o.order_status = 'delivered'
     AND o.order_delivered_customer_date IS NOT NULL)            AS is_delivered,

    -- Positive = days after the estimate, negative = days early.
    CASE WHEN o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
         THEN o.order_delivered_customer_date::date
            - o.order_estimated_delivery_date::date
    END                                                          AS days_late,

    CASE WHEN o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
         THEN o.order_delivered_customer_date::date
            > o.order_estimated_delivery_date::date
    END                                                          AS is_late
FROM orders o
LEFT JOIN customers c ON c.customer_id = o.customer_id;
