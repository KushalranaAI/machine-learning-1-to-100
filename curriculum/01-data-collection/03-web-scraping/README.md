---
title: Web Scraping
order: 3
summary: Extract data from HTML pages with requests + BeautifulSoup - selectors, pagination, cleaning, politeness (robots.txt, delays) and the legal/ethical rules.
level: intermediate
minutes: 45
why: When there is no API or download, the data is often sitting on web pages - prices, reviews, listings, job posts. Scraping turns those pages into a dataset.
background: Web crawlers appeared in 1993 to index the early web. The robots.txt convention (1994) lets sites say what bots may access; it became an IETF standard (RFC 9309) in 2022.
use_cases:
  - Price monitoring and competitor analysis
  - Collecting reviews for sentiment analysis
  - Real-estate or job-listing datasets
  - Building text corpora for NLP
code:
  - curriculum/01-data-collection/03-web-scraping/demo.py
demo: demo.py
pros:
  - Works when no API exists
  - Any visible data can become a dataset
cons:
  - Fragile - breaks when the HTML changes
  - Legal and ethical limits (terms of service, copyright, personal data)
  - JavaScript-rendered pages need a real browser (Playwright/Selenium)
workflow: |
  flowchart LR
    A[Check robots.txt & terms] --> B[Inspect page HTML in DevTools]
    B --> C[Pick CSS selectors]
    C --> D[Fetch page politely: UA, delay]
    D --> E[Parse with BeautifulSoup]
    E --> F[Clean: prices, ratings, dates]
    F --> G{Next page link?}
    G -->|yes| D
    G -->|no| H[Save CSV + source URL + date]
---

## How a page is structured
HTML is a tree of tags. Each product on a listing page usually shares the same structure:
```html
<article class="product">
  <h3><a href="book_3.html" title="Sharp Objects">Sharp Objects</a></h3>
  <p class="star-rating Four"></p>
  <p class="price">£47.82</p>
</article>
```
Open your browser's **DevTools (F12) → Inspect** to find the repeating container and the fields inside it.

## BeautifulSoup + CSS selectors
```python
soup = BeautifulSoup(html, "html.parser")
for card in soup.select("article.product"):          # every product container
    title = card.select_one("h3 a")["title"]          # attribute, not the (truncated) text
    price = float(card.select_one(".price").text.lstrip("£"))
    rating = card.select_one(".star-rating")["class"][1]   # "Four" → 4
```
Useful selectors: `tag`, `.class`, `#id`, `parent > child`, `a[href]`, `li.next a`.

## Pagination
Follow the "next" link until there isn't one (`urljoin` turns relative links into absolute URLs). Set a maximum
page count so a bug can't crawl forever.

## Be a good citizen (and stay legal)
1. **Read the terms of service** - some sites forbid scraping; respect that.
2. **Check robots.txt** with `urllib.robotparser` and obey `Disallow` and `Crawl-delay`.
3. **Identify yourself** with a User-Agent, add a **delay** between requests, cache pages while developing.
4. **Don't collect personal data** without a legal basis (GDPR/CCPA) and don't republish copyrighted content.
5. **Prefer an API** if one exists.

## Dynamic (JavaScript) pages
If `requests.get(url).text` doesn't contain the data you see in the browser, the page builds it with JavaScript.
Options: find the JSON API the page calls (DevTools → Network → XHR) - best - or drive a real browser with
**Playwright** or **Selenium**.

## Cleaning scraped values
Scraped values are strings: strip currency symbols, convert "Four" → 4, parse "In stock (19 available)" → 19,
normalise whitespace, de-duplicate, and **store the source URL and scrape time** with every row.

## Exercises
1. Run the demo on the bundled site, then with `--live` against books.toscrape.com (a site built for practice).
2. Visit each book's detail page and add its description and UPC.
3. Make the scraper resumable: skip URLs already in your CSV.
