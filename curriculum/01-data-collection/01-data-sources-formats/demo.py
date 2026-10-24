"""Write the same table in several formats, read it back, and compare size, speed and type fidelity."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200_000
df = pd.DataFrame({
    "order_id": np.arange(n),
    "zip_code": rng.integers(0, 99999, n).astype(str).astype(object),
    "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "category": rng.choice(["books", "games", "garden", "toys"], n),
    "price": rng.gamma(2.0, 20.0, n).round(2),
})
df["zip_code"] = df["zip_code"].str.zfill(5)   # "00501" - leading zeros matter!

tmp = Path(tempfile.mkdtemp())
writers = {
    "csv": lambda p: df.to_csv(p, index=False),
    "json": lambda p: df.to_json(p, orient="records", date_format="iso"),
    "jsonl": lambda p: df.to_json(p, orient="records", lines=True, date_format="iso"),
}
readers = {
    "csv": lambda p: pd.read_csv(p),
    "json": lambda p: pd.read_json(p),
    "jsonl": lambda p: pd.read_json(p, lines=True),
}
try:
    import pyarrow  # noqa: F401
    writers["parquet"] = lambda p: df.to_parquet(p, index=False)
    readers["parquet"] = lambda p: pd.read_parquet(p)
except ImportError:
    print("(pip install pyarrow to include Parquet)")

print(f"{'format':<9}{'size MB':>9}{'write s':>9}{'read s':>9}   zip_code dtype → first value")
for fmt in writers:
    path = tmp / f"orders.{fmt}"
    t = time.perf_counter(); writers[fmt](path); tw = time.perf_counter() - t
    t = time.perf_counter(); back = readers[fmt](path); tr = time.perf_counter() - t
    print(f"{fmt:<9}{path.stat().st_size / 1e6:>9.2f}{tw:>9.2f}{tr:>9.2f}   {str(back['zip_code'].dtype):<8} → {back['zip_code'].iloc[0]!r}"
          f"   date dtype: {back['order_date'].dtype}")

print("\nText formats (CSV, JSON) turned zip codes into integers (lost leading zeros) and kept dates as strings;"
      "\nParquet stored the real types.")
fixed = pd.read_csv(tmp / "orders.csv", dtype={"zip_code": "string"}, parse_dates=["order_date"])
print("Fixed with dtype=/parse_dates=:", fixed.dtypes.to_dict())

# Nested JSON from an API → flat table
api_response = json.loads('[{"id": 1, "user": {"name": "Ana", "city": "Lima"}, "score": 9.5},'
                          ' {"id": 2, "user": {"name": "Ben", "city": null}, "score": 7.0}]')
print("\njson_normalize on nested API data:\n", pd.json_normalize(api_response))

# Streaming a big CSV in chunks: aggregate without loading everything
totals = None
for chunk in pd.read_csv(tmp / "orders.csv", usecols=["category", "price"], chunksize=50_000):
    part = chunk.groupby("category")["price"].sum()
    totals = part if totals is None else totals.add(part, fill_value=0)
print("\nrevenue per category computed in 50k-row chunks:\n", totals.round(0))
