---
title: Data Sources & File Formats
order: 1
summary: CSV, TSV, JSON, JSON Lines, Parquet, Excel and pickles - how each stores data, how to read and write them with pandas, and which to choose.
level: beginner
minutes: 30
why: The first step of any project is loading data correctly. Wrong delimiters, encodings, date parsing or dtypes silently corrupt everything downstream.
background: CSV dates back to 1970s mainframes and is still the most common exchange format. JSON (2001) became the language of web APIs. Columnar formats like Parquet (2013) were created for big-data analytics.
use_cases:
  - Loading Kaggle competition files (CSV)
  - Reading API responses (JSON) and logs (JSON Lines)
  - Storing large feature tables efficiently (Parquet)
  - Receiving business reports (Excel)
where_used:
  - 01-house-prices
  - 02-titanic-survival
  - 04-spam-detector
demo: demo.py
pros:
  - CSV is universal and human-readable
  - Parquet is compressed, typed and fast for column queries
cons:
  - CSV has no types - everything must be parsed and validated
  - Pickle is unsafe to load from untrusted sources and is Python-version specific
workflow: |
  flowchart LR
    A[File on disk / URL] --> B{Format?}
    B -->|CSV / TSV| C[pd.read_csv sep, dtype, parse_dates, encoding]
    B -->|JSON| D[pd.read_json / json_normalize]
    B -->|Parquet| E[pd.read_parquet]
    B -->|Excel| F[pd.read_excel sheet_name]
    C & D & E & F --> G[DataFrame]
    G --> H[Check: shape, dtypes, head, nulls]
---

## The formats

| Format | Structure | Types kept? | Size / speed | Typical source |
|--------|-----------|-------------|--------------|----------------|
| **CSV / TSV** | rows of delimited text | ✗ (all text) | large, slow | exports, Kaggle |
| **JSON** | nested objects/arrays | partially | large | web APIs |
| **JSON Lines** | one JSON object per line | partially | streamable | logs, events |
| **Parquet** | columnar, compressed | ✓ | small, fast | data lakes, Spark |
| **Excel** | sheets, cells, formatting | partially | medium | business teams |
| **SQL database** | tables with schema | ✓ | indexed queries | applications (lesson 4) |
| **Pickle / joblib** | Python objects | ✓ | fast | saving models - never load untrusted ones |

## Reading CSV correctly
```python
df = pd.read_csv(
    "sales.csv",
    sep=",",                       # ';' is common in Europe, '\t' for TSV
    encoding="utf-8",              # 'latin-1' for old Windows exports
    parse_dates=["order_date"],    # real datetime dtype, not strings
    dtype={"zip_code": "string"},  # keep leading zeros!
    na_values=["", "NA", "?", "-"],# what counts as missing in this file
    thousands=",",
)
```
**Always inspect right after loading:** `df.shape`, `df.dtypes`, `df.head()`, `df.isna().sum()`, `df.describe()`.

Classic traps:
- Zip codes / IDs read as integers lose leading zeros.
- Dates stay as strings → you can't extract month or compute durations.
- A stray `"?"` makes a numeric column `object` dtype.
- Huge files: use `usecols=`, `dtype=` and `chunksize=` to stream.

## JSON and nested data
API responses are nested. `pd.json_normalize` flattens them into columns:
```python
records = [{"id": 1, "user": {"name": "Ana", "city": "Lima"}, "tags": ["a", "b"]}]
pd.json_normalize(records)   # columns: id, tags, user.name, user.city
```

## Parquet for anything big
Parquet stores each column together, compressed and typed. Reading 3 columns out of 200 only touches those 3.
Typical result: **5-10× smaller and much faster** than CSV, and dtypes survive the round-trip.

## Reproducibility habits
- Keep **raw** data read-only; write cleaned data to a new file.
- Record where each file came from (URL, date, version, licence).
- Store large data outside git (or with DVC / Git LFS) - see `.gitignore`.

## Exercises
1. Run the demo and compare file sizes and load times for CSV vs Parquet vs JSON.
2. Make the demo's CSV use `;` and a comma decimal (`3,14`). Which `read_csv` arguments fix it?
3. Load a JSON Lines file of 1 million rows in chunks and count rows per category without loading it all.
