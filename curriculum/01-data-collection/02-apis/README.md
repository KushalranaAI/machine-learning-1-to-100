---
title: Collecting Data from APIs
order: 2
summary: REST APIs, JSON, authentication, pagination, rate limits, retries and caching - a robust API client from scratch.
level: beginner
minutes: 40
why: Most live data (weather, finance, social media, internal company services, LLMs) is served through APIs. A good client handles pagination, errors and rate limits so your dataset is complete and your key doesn't get banned.
background: REST was described by Roy Fielding in 2000. Today almost every service exposes a JSON-over-HTTPS API, usually with API-key or OAuth authentication.
use_cases:
  - Pulling stock prices, weather or exchange rates for forecasting
  - Collecting GitHub/Reddit/news data for NLP
  - Reading data from internal microservices
  - Enriching records (geocoding, company info)
where_used:
  - 01-house-prices
code:
  - mlscratch/download.py:_http_get
demo: demo.py
pros:
  - Structured, documented, usually legal to use within terms
  - Fresh data on demand
cons:
  - Rate limits and quotas
  - Pagination and schema changes must be handled
  - Keys must be kept secret (environment variables, never in git)
workflow: |
  flowchart LR
    A[Read API docs] --> B[Get API key → env variable]
    B --> C[Request page 1]
    C --> D{HTTP status}
    D -->|200| E[Parse JSON, store records]
    D -->|429 / 5xx| F[Wait: back-off, Retry-After] --> C
    E --> G{Next page?}
    G -->|yes| C
    G -->|no| H[Save raw JSON + DataFrame]
---

## HTTP in five minutes
A request = **method** (GET reads, POST creates) + **URL** + **headers** (auth, content type) + optional **body**.
The response has a **status code**:

| Code | Meaning | What your client should do |
|------|---------|----------------------------|
| 200 | OK | parse the body |
| 301/302 | moved | follow the redirect |
| 400 | bad request | fix your parameters |
| 401 / 403 | not authenticated / not allowed | check the key / permissions |
| 404 | not found | skip or stop |
| **429** | **too many requests** | wait (`Retry-After` header) and retry |
| 5xx | server error | retry with exponential back-off |

## Authentication
- **API key** in a header (`Authorization: Bearer <key>` or `X-API-Key`). Read it from an environment variable:
  `os.environ["MY_API_KEY"]`. Never hard-code or commit it.
- **OAuth 2** for user data (Google, GitHub apps): exchange credentials for a short-lived token.

## Pagination patterns
APIs return data in pages. Three common styles:
1. **Page number**: `?page=2&per_page=100` - loop until a page is empty.
2. **Offset/limit**: `?offset=200&limit=100`.
3. **Cursor / link header**: the response tells you the next URL (`Link: <...>; rel="next"` on GitHub). Most robust.

## A robust client (pattern used in `demo.py`)
```python
def get_json(session, url, params=None, max_retries=5):
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))   # exponential back-off
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json(), r
    raise RuntimeError(f"gave up on {url}")
```
Plus: reuse a `requests.Session` (keeps connections open), set a timeout on every call, **cache** responses to disk
while developing so you don't re-download, and save the **raw JSON** before transforming it.

## From JSON to a DataFrame
Keep only the fields you need, flatten nested ones (`pd.json_normalize`), convert timestamps, and record the
collection time - data from APIs changes, so note *when* you pulled it.

## Exercises
1. Run the demo; it pages through GitHub's public API (no key needed, 60 requests/hour). Add `per_page` and watch the number of calls change.
2. Add a disk cache: hash the URL + params, store the JSON, and return it on the next run.
3. Use an API you care about (e.g. Open-Meteo weather, no key) and build a daily table for one city.
