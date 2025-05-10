# Placeholder for scraper.py
"""
Generic web scraper to json file

- Configutre the START_URLS are the lists of pages to fetch.
- Configure FIELDS with:
    1. selector: CSS selector
    2. attributeL HTML attribute to extract test
    3. multiple: True to return the list, False to return the first match

- Run and get a JSON file of the all extracted records.
"""

import json
import logging
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# -------------------------------Configuration-----------------------------------
# URL you want to scrape

START_URLS = [
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://www.wikihow.com/Main-Page",
]


Fields = {
    "page_title": {"selector": "title", "attribute": "text", "multiple": False},
    "heading": {
        "selector": "h1, h2, h3, h4, h5, h6",
        "attribute": "text",
        "multiple": False,
    },
    "links": {"selector": "a", "attribute": "href", "multiple": True},
    "images": {"selector": "img", "attribute": "src", "multiple": True},
    "paragraphs": {"selector": "p", "attribute": "text", "multiple": True},
    # add more field if you want
}

OUTPUT_JSON_PATH = os.path.join("Scrape_dataset", "simple_scraped_data.json")


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GenericScraper/1.0)"}


# ------------------------------------------------logging setup--------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# ----------------------------------------------Scraping Functions-------------------------------------------------------


def fetch_html(url: str) -> str:
    """
    Fetch HTML content from a given URL.
    """
    logging.info(f"-> Fetching URL: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_fields(html: str, base_url: str) -> dict:
    """Parses HTML and extracts data per the FIELDS config"""
    soup = BeautifulSoup(html, "html.parser")
    record = {}

    for field_name, field_Config in Fields.items():
        els = soup.select(field_Config["selector"])
        if not els:
            record[field_name] = [] if field_Config["multiple"] else None
            continue

        def get_val(el):
            if field_Config["attribute"] == "text":
                return el.get_text(strip=True)
            else:
                val = el.get(field_Config["attribute"])
                # convert relative URLs to absolute if it link/src
                if field_Config["attribute"] in ("href", "src") and val:
                    return urljoin(base_url, val)
                return val

        if field_Config["multiple"]:
            record[field_name] = [get_val(el) for el in els]
        else:
            record[field_name] = get_val(els[0])

    return record


def save_json(data: list, path: str):
    """Save list of dicts as prettu printy JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"send {len(data)} records to {path}")


# ------------------------------------ Main Script-----------------------------------


def main():
    all_data = []
    for url in START_URLS:
        try:
            html = fetch_html(url)
            rec = extract_fields(html, url)
            rec["url"] = url
            all_data.append(rec)
        except Exception as err:
            logging.error(f"Error processing {url}: {err}")
    if all_data:
        save_json(all_data, OUTPUT_JSON_PATH)
    else:
        logging.warning("No data scraped-check your URL and selector.")


if __name__ == "__main__":
    main()
