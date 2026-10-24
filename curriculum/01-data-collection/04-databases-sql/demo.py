"""Build a small shop database in SQLite, then turn raw tables into an ML feature table with SQL."""

import sqlite3

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
con = sqlite3.connect(":memory:")

customers = pd.DataFrame({
    "customer_id": range(1, 201),
    "country": rng.choice(["PE", "IN", "US", "DE"], 200),
    "signup_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, 200), unit="D"),
})
n_orders = 1500
orders = pd.DataFrame({
    "order_id": range(1, n_orders + 1),
    "customer_id": rng.integers(1, 201, n_orders),
    "order_date": pd.Timestamp("2024-01-15") + pd.to_timedelta(rng.integers(0, 520, n_orders), unit="D"),
    "amount": rng.gamma(2, 30, n_orders).round(2),
})
tickets = pd.DataFrame({"customer_id": rng.integers(1, 201, 300),
                        "opened": pd.Timestamp("2024-02-01") + pd.to_timedelta(rng.integers(0, 480, 300), unit="D")})
for name, df in [("customers", customers), ("orders", orders), ("tickets", tickets)]:
    df.to_sql(name, con, index=False)
print("tables:", [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")])

CUTOFF = "2025-03-01"
features = pd.read_sql("""
    WITH o AS (SELECT * FROM orders WHERE order_date < :cutoff),            -- only the past!
         t AS (SELECT customer_id, COUNT(*) AS n_tickets FROM tickets
               WHERE opened < :cutoff GROUP BY customer_id)
    SELECT c.customer_id,
           c.country,
           COUNT(o.order_id)                                         AS frequency,
           ROUND(COALESCE(SUM(o.amount), 0), 2)                      AS monetary,
           CAST(julianday(:cutoff) - julianday(MAX(o.order_date)) AS INT) AS recency_days,
           COALESCE(t.n_tickets, 0)                                  AS n_tickets
    FROM customers c
    LEFT JOIN o ON o.customer_id = c.customer_id
    LEFT JOIN t ON t.customer_id = c.customer_id
    WHERE c.signup_date < :cutoff
    GROUP BY c.customer_id
    ORDER BY monetary DESC
""", con, params={"cutoff": CUTOFF})
print(f"\nRFM feature table (cut-off {CUTOFF}): {features.shape}\n", features.head(8).to_string(index=False))

# Label: did the customer order in the 90 days AFTER the cut-off? (what we would predict)
labels = pd.read_sql("""
    SELECT customer_id, 1 AS ordered_next_90d FROM orders
    WHERE order_date >= :cutoff AND order_date < date(:cutoff, '+90 days')
    GROUP BY customer_id
""", con, params={"cutoff": CUTOFF})
data = features.merge(labels, on="customer_id", how="left").fillna({"ordered_next_90d": 0})
print("\nlabel balance:", data["ordered_next_90d"].value_counts().to_dict())

# Window function: running spend per customer, without collapsing rows
running = pd.read_sql("""
    SELECT customer_id, order_date, amount,
           SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total,
           RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC)     AS amount_rank
    FROM orders WHERE customer_id = 7 ORDER BY order_date LIMIT 6
""", con)
print("\nwindow functions for customer 7:\n", running.to_string(index=False))

# SQL injection: why we always pass parameters
evil = "PE' OR '1'='1"
unsafe = con.execute(f"SELECT COUNT(*) FROM customers WHERE country = '{evil}'").fetchone()[0]
safe = con.execute("SELECT COUNT(*) FROM customers WHERE country = ?", (evil,)).fetchone()[0]
print(f"\nf-string query with malicious input returned {unsafe} rows (all!); parameterised query returned {safe}.")
