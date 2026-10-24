---
title: Data Handling & Wrangling
icon: "🧹"
description: Turn raw, messy tables into clean, tidy, validated data - pandas, missing values, outliers, reshaping, merging and data-quality checks.
---

> "Data scientists spend 80% of their time cleaning data." The number is debated - the reality isn't.

```mermaid
flowchart LR
  R[(Raw data)] --> L[Load & inspect]
  L --> T[Fix types: numbers, dates, categories]
  T --> M[Missing values]
  M --> D[Duplicates & inconsistent labels]
  D --> O[Outliers]
  O --> S[Reshape & merge into one tidy table]
  S --> V[Validate: schema, ranges, uniqueness]
  V --> C[(Clean data → features)]
```

**Tidy data** (Hadley Wickham, 2014): every *column* is one variable, every *row* is one observation, every *cell* is one value.
Almost every ML library expects a tidy table.

All lessons in this part use `data/messy_customers.csv`, a deliberately dirty dataset (mixed date formats,
"N/A"/"?" codes, duplicates, impossible ages, `$`-formatted incomes, heights in two units, `yes/Y/1` labels).
