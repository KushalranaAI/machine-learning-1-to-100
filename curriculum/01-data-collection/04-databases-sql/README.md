---
title: Databases & SQL
order: 4
summary: Pull training data out of relational databases - SELECT, WHERE, JOIN, GROUP BY, window functions - and load it into pandas safely.
level: beginner
minutes: 45
why: Company data lives in databases. Data scientists write SQL every day to build training tables - joining customers, orders and events, and aggregating them into features.
background: Codd's relational model (1970) and SQL (1974, IBM) still run most businesses. Analytical databases (Postgres, BigQuery, Snowflake, DuckDB) speak the same language.
use_cases:
  - Building a churn dataset from customers + orders + support tickets
  - Aggregating event logs into per-user features
  - Joining labels with features from different tables
where_used:
  - 03-customer-segmentation
demo: demo.py
pros:
  - The database does the heavy lifting (filtering, joins, aggregation) close to the data
  - Declarative, reproducible queries
cons:
  - Easy to create target leakage with time-unaware joins
  - Never build SQL with string formatting from user input (SQL injection)
workflow: |
  flowchart LR
    A[(customers)] --> J[JOIN on customer_id]
    B[(orders)] --> J
    C[(tickets)] --> J
    J --> G[GROUP BY customer → features]
    G --> F[Filter by cut-off date - no leakage]
    F --> P[pd.read_sql → DataFrame]
---

## The core of SQL for ML
```sql
SELECT c.customer_id,
       c.country,
       COUNT(o.order_id)              AS n_orders,      -- aggregate per customer
       SUM(o.amount)                  AS total_spent,
       MAX(o.order_date)              AS last_order
FROM customers c
LEFT JOIN orders o                                      -- keep customers with zero orders
       ON o.customer_id = c.customer_id
      AND o.order_date < '2025-06-01'                   -- only data known before the cut-off!
WHERE c.signup_date < '2025-06-01'
GROUP BY c.customer_id, c.country
HAVING COUNT(o.order_id) >= 0
ORDER BY total_spent DESC;
```

| Clause | Purpose |
|--------|---------|
| `SELECT ... FROM` | columns and table |
| `WHERE` | filter rows **before** grouping |
| `JOIN ... ON` | combine tables (`INNER` = matches only, `LEFT` = keep all left rows) |
| `GROUP BY` + `COUNT/SUM/AVG/MAX` | one row per group → features |
| `HAVING` | filter groups **after** grouping |
| `ORDER BY`, `LIMIT` | sort and cut |
| Window functions `OVER (PARTITION BY ... ORDER BY ...)` | running totals, ranks, lags - without collapsing rows |

## Avoiding time leakage
To predict whether a customer churns in June, features may only use data from **before June 1st**. Put the
cut-off in the JOIN/WHERE clauses. Leakage here gives fantastic validation scores and a useless model.

## From SQL to pandas safely
```python
import sqlite3, pandas as pd
con = sqlite3.connect("shop.db")
df = pd.read_sql("SELECT * FROM orders WHERE country = ?", con, params=("PE",))   # parameterised: no injection
```
Use **parameters** (`?` / `%s`) - never f-strings - for values.

## Exercises
1. Run the demo, then add a feature "number of support tickets in the last 90 days before the cut-off".
2. Rewrite the RFM query with a window function to compute each customer's average days between orders.
3. Point `pd.read_sql` at a Postgres database with SQLAlchemy (`create_engine("postgresql://...")`).
