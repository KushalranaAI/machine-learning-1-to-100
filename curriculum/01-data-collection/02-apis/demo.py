"""A polite, robust API client: sessions, timeouts, retries with back-off, pagination and caching.

Uses GitHub's public REST API (no key needed). Falls back to a bundled sample if you are offline.
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    raise SystemExit("pip install requests")

CACHE = Path("outputs/api_cache")
CACHE.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.headers["User-Agent"] = "ml-1-to-100-course"
calls = {"network": 0, "cache": 0}


def get_json(url, params=None, max_retries=4):
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
    cached = CACHE / f"{key}.json"
    if cached.exists():                                   # cache hit: no network call
        calls["cache"] += 1
        return json.loads(cached.read_text())
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=20)
        calls["network"] += 1
        if resp.status_code in (429, 500, 502, 503, 504) or (resp.status_code == 403 and "rate limit" in resp.text.lower()):
            wait = float(resp.headers.get("Retry-After", 2**attempt))
            print(f"  {resp.status_code} → retrying in {wait:.0f}s")
            time.sleep(min(wait, 10))
            continue
        resp.raise_for_status()
        data = resp.json()
        cached.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"gave up on {url}")


def fetch_repos(query, pages=3, per_page=30):
    """Page-number pagination over GitHub's repository search."""
    rows = []
    for page in range(1, pages + 1):
        data = get_json("https://api.github.com/search/repositories",
                        {"q": query, "sort": "stars", "per_page": per_page, "page": page})
        items = data.get("items", [])
        print(f"page {page}: {len(items)} repos")
        if not items:
            break
        rows.extend(items)
    return rows


online = True
try:
    repos = fetch_repos("machine learning language:python")
except Exception as exc:
    online = False
    print(f"Offline or blocked ({exc.__class__.__name__}); using a tiny built-in sample instead.")
    repos = [{"full_name": "scikit-learn/scikit-learn", "stargazers_count": 60000, "forks_count": 25000,
              "language": "Python", "created_at": "2010-08-17T09:43:38Z", "owner": {"type": "Organization"}, "license": {"spdx_id": "BSD-3-Clause"}}]

df = pd.json_normalize(repos)[["full_name", "stargazers_count", "forks_count", "created_at", "owner.type", "license.spdx_id"]]
df["created_at"] = pd.to_datetime(df["created_at"])
df["age_years"] = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.days / 365.25
df["stars_per_year"] = (df["stargazers_count"] / df["age_years"]).round(0)
print(f"\ncollected {len(df)} repositories  (network calls: {calls['network']}, cache hits: {calls['cache']})")
print(df.sort_values("stars_per_year", ascending=False).head(10).to_string(index=False))
if online:
    print("\nRun the script again: everything now comes from the cache - zero network calls.")
