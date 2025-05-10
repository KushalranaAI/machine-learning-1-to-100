#!/usr/bin/env python3
"""
Advanced Selenium Web Scraper → JSON Export

Features:
 - Headless Chrome with custom UA & optional proxy
 - Explicit waits for JS-rendered content
 - Auto-scroll to load lazy elements
 - Screenshot capture on error
 - Pagination support (click "Next" links)
 - Configurable extraction via CSS selectors
 - Robust logging & retry logic
"""

import json
import logging
import os
import time
from urllib.parse import urljoin

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ------------------------ CONFIGURATION ------------------------

# 1) Pages to start from (can be listing pages with pagination)
START_URLS = [
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://www.wikihow.com/Main-Page",
]

# 2) Extraction rules
FIELDS = {
    "page_title": {"selector": "title", "attr": "text", "multiple": False},
    "heading": {
        "selector": "h1, h2, h3, h4, h5, h6",
        "attr": "text",
        "multiple": False,
    },
    "links": {"selector": "a", "attr": "href", "multiple": True},
    "images": {"selector": "img", "attr": "src", "multiple": True},
    "paragraphs": {"selector": "p", "attr": "text", "multiple": True},
    # add more field if you want
}

# 3) Pagination ("Next" button) CSS selector, or None
PAGINATION_NEXT_SELECTOR = "a.next"

# 4) Output path
OUTPUT_JSON_PATH = os.path.join("Scrape_dataset", "selenium_scraped_data.json")

# 5) Browser settings
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)
PROXY = None  # e.g. "123.45.67.89:3128" or None

# ------------------------ LOGGING SETUP ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ------------------------ SCRAPER CLASS ------------------------


class SeleniumScraper:
    def __init__(self):
        chrome_opts = Options()
        chrome_opts.headless = True
        chrome_opts.add_argument(f"--user-agent={USER_AGENT}")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--window-size=1920,1080")
        chrome_opts.add_argument("--no-sandbox")
        # chrome_opts.add_argument("--incognito")
        if PROXY:
            chrome_opts.add_argument(f"--proxy-server={PROXY}")

        # initialize driver
        # Use Service to point to the driver binary
        service = Service(ChromeDriverManager().install())
        # faster page load
        self.driver = webdriver.Chrome(service=service, options=chrome_opts)
        self.driver.set_page_load_timeout(30)

    def fetch(self, url: str):
        logging.info(f"→ Loading: {url}")
        try:
            self.driver.get(url)
        except WebDriverException as e:
            logging.error(f"Page load error: {e}. Retrying once…")
            time.sleep(5)
            self.driver.get(url)

    def wait_for(self, selector: str, timeout=15):
        """Wait until at least one element matches selector."""
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )

    def auto_scroll(self, pause=1.0, max_scrolls=10):
        """Scroll to bottom to trigger lazy-loads."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for _ in range(max_scrolls):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(pause)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def extract(self, base_url: str) -> dict:
        """Extract all configured fields from the current page."""
        record = {}
        for name, cfg in FIELDS.items():
            els = self.driver.find_elements(By.CSS_SELECTOR, cfg["selector"])
            vals = []
            for el in els:
                if cfg["attr"] == "text":
                    vals.append(el.text.strip())
                else:
                    raw = el.get_attribute(cfg["attr"])
                    # make relative URLs absolute
                    if cfg["attr"] in ("href", "src") and raw:
                        raw = urljoin(base_url, raw)
                    vals.append(raw)
            if not els:
                record[name] = [] if cfg["multiple"] else None
            else:
                record[name] = vals if cfg["multiple"] else vals[0]
        # always include URL
        record["url"] = self.driver.current_url
        return record

    def capture_screenshot(self, name="error.png"):
        path = os.path.join("data", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.driver.save_screenshot(path)
        logging.info(f"↪ Screenshot saved: {path}")

    def close(self):
        self.driver.quit()


# ------------------------ MAIN WORKFLOW ------------------------


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    scraper = SeleniumScraper()
    all_data = []

    try:
        for start_url in START_URLS:
            next_url = start_url
            while next_url:
                try:
                    scraper.fetch(next_url)
                    # wait for one of your key selectors to ensure JS ran
                    first_selector = next(iter(FIELDS.values()))["selector"]
                    scraper.wait_for(first_selector)

                    # optional: scroll to load all lazy content
                    scraper.auto_scroll()

                    # extract data
                    data = scraper.extract(next_url)
                    all_data.append(data)
                    logging.info(f"✔ Extracted data from {data['url']}")

                    # handle pagination
                    if PAGINATION_NEXT_SELECTOR:
                        try:
                            nxt = scraper.driver.find_element(
                                By.CSS_SELECTOR, PAGINATION_NEXT_SELECTOR
                            )
                            next_url = nxt.get_attribute("href")
                        except NoSuchElementException:
                            next_url = None
                    else:
                        next_url = None

                except Exception as e:
                    logging.error(f"✖ Failed on {next_url}: {e}")
                    scraper.capture_screenshot("scrape_error.png")
                    break  # stop pagination loop for this start_url

    finally:
        scraper.close()

    # save JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logging.info(f"■ Saved {len(all_data)} records to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
