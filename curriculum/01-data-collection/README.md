---
title: Data Collection
icon: "📥"
description: Where data comes from and how to get it into Python - files, APIs, web scraping, databases and public dataset hubs like Kaggle.
---

Every ML project starts with a question **and the data to answer it**. In industry, collecting and preparing data
takes most of the time - models are the easy part.

```mermaid
flowchart LR
  Q[Business question] --> S{Where is the data?}
  S --> F[Files: CSV, JSON, Parquet, Excel]
  S --> A[APIs: REST / JSON]
  S --> W[Websites: scraping]
  S --> D[Databases: SQL]
  S --> P[Public hubs: Kaggle, UCI, OpenML]
  F & A & W & D & P --> R[(Raw data store)]
  R --> N[Next: wrangling & cleaning]
```

Questions to ask about any data source:
1. **Is it representative** of the cases the model will see in production?
2. **Is it legal and ethical** to use (licence, terms of service, personal data / GDPR)?
3. **How fresh** is it, and can you get it again tomorrow (reproducibility)?
4. **How was it labelled**, and how noisy are the labels?
