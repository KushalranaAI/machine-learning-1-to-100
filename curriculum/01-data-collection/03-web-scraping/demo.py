"""Scrape a paginated catalogue with requests + BeautifulSoup, politely.

Default: the small bundled site in ./site (works offline).
--live : https://books.toscrape.com - a sandbox site made for scraping practice.
"""

import argparse
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
HERE = Path(__file__).resolve().parent


def fetch(url: str) -> str:
    if url.startswith("file://"):
        return Path(url[7:]).read_text(encoding="utf-8")
    import requests

    resp = requests.get(url, headers={"User-Agent": "ml-1-to-100-course (learning project)"}, timeout=20)
    resp.raise_for_status()
    return resp.text


def allowed(start_url: str, path: str, live: bool) -> tuple[bool, float]:
    rp = urllib.robotparser.RobotFileParser()
    if live:
        rp.set_url(urljoin(start_url, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            return True, 1.0
    else:
        rp.parse((HERE / "site" / "robots.txt").read_text().splitlines())
    return rp.can_fetch("*", path), float(rp.crawl_delay("*") or 1.0)


def parse_page(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("article.product, article.product_pod"):
        link = card.select_one("h3 a")
        price_text = card.select_one(".price, .price_color").get_text(strip=True)
        rows.append({
            "title": link["title"],                                         # full title (text may be truncated)
            "price_gbp": float(re.sub(r"[^\d.]", "", price_text)),           # "£51.77" → 51.77
            "rating": RATINGS[card.select_one(".star-rating")["class"][1]],  # "star-rating Three" → 3
            "in_stock": "in stock" in card.select_one(".availability").get_text().lower(),
            "url": urljoin(base_url, link["href"]),
        })
    nxt = soup.select_one("li.next a")
    return rows, (urljoin(base_url, nxt["href"]) if nxt else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-pages", type=int, default=3)
    args = ap.parse_args()

    url = "https://books.toscrape.com/catalogue/page-1.html" if args.live else (HERE / "site" / "page-1.html").as_uri()
    ok, delay = allowed(url, "/catalogue/" if args.live else "/page-1.html", args.live)
    print(f"robots.txt allows us: {ok}; crawl delay {delay}s")
    print("robots.txt check for /admin/ on the demo site:", allowed(url, "/admin/", False)[0])
    if not ok:
        raise SystemExit("Not allowed by robots.txt - stopping.")

    rows, pages = [], 0
    while url and pages < args.max_pages:
        page_rows, url = parse_page(fetch(url), url)
        rows += page_rows
        pages += 1
        print(f"page {pages}: {len(page_rows)} products")
        if url and args.live:
            time.sleep(delay)            # be polite

    df = pd.DataFrame(rows).drop_duplicates("url")
    df["scraped_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    print(f"\n{len(df)} products scraped\n", df.drop(columns=["url", "scraped_at"]).head(8).to_string(index=False))
    print("\naverage price by rating:\n", df.groupby("rating")["price_gbp"].agg(["count", "mean"]).round(2))
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/books.csv", index=False)
    print("\nsaved outputs/books.csv")
