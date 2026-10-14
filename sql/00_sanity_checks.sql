-- 00_sanity_checks.sql
-- Verifies the raw load from src/load_to_postgres.py. Run after every load:
--   docker exec -i olist-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/00_sanity_checks.sql
--
-- These checks confirm the load is faithful and describe the data's gaps.
-- They do not fix anything: raw tables stay exactly as the CSVs.


-- 1. Row counts ---------------------------------------------------------------
-- Question: did every CSV row arrive, and nothing extra?
-- expected_rows = data rows in each CSV (counted with pandas, not `wc -l`,
-- because review comments contain line breaks inside quoted fields).

SELECT t.table_name, t.actual_rows, t.expected_rows,
       t.actual_rows = t.expected_rows AS ok
FROM (
              SELECT 'customers'                    AS table_name, (SELECT COUNT(*) FROM customers)                    AS actual_rows,   99441 AS expected_rows
    UNION ALL SELECT 'geolocation',                   (SELECT COUNT(*) FROM geolocation),                   1000163
    UNION ALL SELECT 'order_items',                   (SELECT COUNT(*) FROM order_items),                    112650
    UNION ALL SELECT 'order_payments',                (SELECT COUNT(*) FROM order_payments),                 103886
    UNION ALL SELECT 'order_reviews',                 (SELECT COUNT(*) FROM order_reviews),                   99224
    UNION ALL SELECT 'orders',                        (SELECT COUNT(*) FROM orders),                          99441
    UNION ALL SELECT 'products',                      (SELECT COUNT(*) FROM products),                        32951
    UNION ALL SELECT 'sellers',                       (SELECT COUNT(*) FROM sellers),                          3095
    UNION ALL SELECT 'product_category_translation',  (SELECT COUNT(*) FROM product_category_translation),      71
) AS t
ORDER BY t.table_name;


-- 2. Date ranges --------------------------------------------------------------
-- Question: were timestamps parsed correctly? The dataset covers 2016-2018;
-- anything far outside that points to a parsing problem.

SELECT 'order_purchase_timestamp'      AS column_name, MIN(order_purchase_timestamp)      AS min_value, MAX(order_purchase_timestamp)      AS max_value FROM orders
UNION ALL SELECT 'order_approved_at',             MIN(order_approved_at),             MAX(order_approved_at)             FROM orders
UNION ALL SELECT 'order_delivered_carrier_date',  MIN(order_delivered_carrier_date),  MAX(order_delivered_carrier_date)  FROM orders
UNION ALL SELECT 'order_delivered_customer_date', MIN(order_delivered_customer_date), MAX(order_delivered_customer_date) FROM orders
UNION ALL SELECT 'order_estimated_delivery_date', MIN(order_estimated_delivery_date), MAX(order_estimated_delivery_date) FROM orders
UNION ALL SELECT 'shipping_limit_date',           MIN(shipping_limit_date),           MAX(shipping_limit_date)           FROM order_items
UNION ALL SELECT 'review_creation_date',          MIN(review_creation_date),          MAX(review_creation_date)          FROM order_reviews;


-- 3. NULLs in key date columns --------------------------------------------------
-- Question: how many orders are missing each date that "late delivery" depends on?

SELECT COUNT(*)                                                         AS total_orders,
       COUNT(*) FILTER (WHERE order_purchase_timestamp      IS NULL)    AS null_purchase,
       COUNT(*) FILTER (WHERE order_approved_at             IS NULL)    AS null_approved,
       COUNT(*) FILTER (WHERE order_delivered_carrier_date  IS NULL)    AS null_carrier,
       COUNT(*) FILTER (WHERE order_delivered_customer_date IS NULL)    AS null_delivered,
       COUNT(*) FILTER (WHERE order_estimated_delivery_date IS NULL)    AS null_estimated
FROM orders;

-- Question: are the missing dates explained by order_status (e.g. a canceled
-- order never gets delivered), or do some "delivered" orders lack a delivery date?

SELECT order_status,
       COUNT(*)                                                         AS orders,
       COUNT(*) FILTER (WHERE order_approved_at             IS NULL)    AS null_approved,
       COUNT(*) FILTER (WHERE order_delivered_carrier_date  IS NULL)    AS null_carrier,
       COUNT(*) FILTER (WHERE order_delivered_customer_date IS NULL)    AS null_delivered
FROM orders
GROUP BY order_status
ORDER BY orders DESC;


-- 4a. Orphan foreign keys (child row points to a parent that does not exist) ----
-- Question: which joins in the Kaggle schema would silently drop rows?
-- Foreign keys are deliberately not enforced in the raw tables, so this is
-- where broken links show up.

SELECT 'order_items.order_id -> orders'      AS relationship, COUNT(*) AS orphans
FROM order_items c WHERE NOT EXISTS (SELECT 1 FROM orders p WHERE p.order_id = c.order_id)
UNION ALL
SELECT 'order_items.product_id -> products', COUNT(*)
FROM order_items c WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = c.product_id)
UNION ALL
SELECT 'order_items.seller_id -> sellers', COUNT(*)
FROM order_items c WHERE NOT EXISTS (SELECT 1 FROM sellers p WHERE p.seller_id = c.seller_id)
UNION ALL
SELECT 'order_payments.order_id -> orders', COUNT(*)
FROM order_payments c WHERE NOT EXISTS (SELECT 1 FROM orders p WHERE p.order_id = c.order_id)
UNION ALL
SELECT 'order_reviews.order_id -> orders', COUNT(*)
FROM order_reviews c WHERE NOT EXISTS (SELECT 1 FROM orders p WHERE p.order_id = c.order_id)
UNION ALL
SELECT 'orders.customer_id -> customers', COUNT(*)
FROM orders c WHERE NOT EXISTS (SELECT 1 FROM customers p WHERE p.customer_id = c.customer_id)
UNION ALL
SELECT 'products.category -> translation', COUNT(*)
FROM products c WHERE c.product_category_name IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM product_category_translation p WHERE p.product_category_name = c.product_category_name)
-- Zip prefixes are counted as distinct values: geolocation has many rows per
-- prefix, so the question is "which prefixes have no coordinates at all".
UNION ALL
SELECT 'customers.zip -> geolocation (distinct zips)', COUNT(DISTINCT c.customer_zip_code_prefix)
FROM customers c WHERE NOT EXISTS (SELECT 1 FROM geolocation p WHERE p.geolocation_zip_code_prefix = c.customer_zip_code_prefix)
UNION ALL
SELECT 'sellers.zip -> geolocation (distinct zips)', COUNT(DISTINCT c.seller_zip_code_prefix)
FROM sellers c WHERE NOT EXISTS (SELECT 1 FROM geolocation p WHERE p.geolocation_zip_code_prefix = c.seller_zip_code_prefix);


-- 4b. Childless parents (order with no rows in a child table) -----------------
-- Question: how many orders would disappear from an INNER JOIN to items,
-- payments, or reviews?

SELECT 'orders without items'    AS check_name, COUNT(*) AS orders
FROM orders o WHERE NOT EXISTS (SELECT 1 FROM order_items   c WHERE c.order_id = o.order_id)
UNION ALL
SELECT 'orders without payments', COUNT(*)
FROM orders o WHERE NOT EXISTS (SELECT 1 FROM order_payments c WHERE c.order_id = o.order_id)
UNION ALL
SELECT 'orders without reviews',  COUNT(*)
FROM orders o WHERE NOT EXISTS (SELECT 1 FROM order_reviews c WHERE c.order_id = o.order_id);
