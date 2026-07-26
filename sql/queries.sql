-- Analysis queries for the US retail-sales series.
-- The data lives in a table `retail_sales(date TEXT, sales REAL)` where `date`
-- is 'YYYY-MM-DD' (month start) and `sales` is $ millions.
-- Each query is named with a `-- name:` header so sql_analysis.py can load them.

-- name: yearly_totals
-- Total sales per calendar year (in $ billions).
SELECT strftime('%Y', date)              AS year,
       ROUND(SUM(sales) / 1000.0, 1)     AS total_sales_billions,
       COUNT(*)                          AS months
FROM retail_sales
GROUP BY year
ORDER BY year;

-- name: seasonal_index
-- Average sales by calendar month, indexed to 100 = an average month.
WITH monthly AS (
    SELECT CAST(strftime('%m', date) AS INTEGER) AS month,
           AVG(sales)                            AS avg_sales
    FROM retail_sales
    GROUP BY month
)
SELECT month,
       ROUND(avg_sales / (SELECT AVG(avg_sales) FROM monthly) * 100, 1) AS seasonal_index
FROM monthly
ORDER BY month;

-- name: top_months
-- The five highest-sales months on record.
SELECT date,
       ROUND(sales / 1000.0, 1) AS sales_billions
FROM retail_sales
ORDER BY sales DESC
LIMIT 5;

-- name: yoy_growth
-- Year-over-year growth of annual totals, using a window function (LAG).
-- Only complete (12-month) years are compared, so a partial current year
-- doesn't produce a misleading drop.
WITH yearly AS (
    SELECT strftime('%Y', date) AS year,
           SUM(sales)           AS total
    FROM retail_sales
    GROUP BY year
    HAVING COUNT(*) = 12
)
SELECT year,
       ROUND((total - LAG(total) OVER (ORDER BY year))
             * 100.0 / LAG(total) OVER (ORDER BY year), 1) AS yoy_growth_pct
FROM yearly
ORDER BY year;
