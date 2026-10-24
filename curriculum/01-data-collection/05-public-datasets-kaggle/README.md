---
title: Public Datasets & the Kaggle API
order: 5
summary: Find and download datasets from Kaggle, UCI, OpenML, Hugging Face and government portals - with the Kaggle CLI, licences, and reproducible download scripts.
level: beginner
minutes: 30
why: Public datasets let you learn, benchmark and prototype without collecting data yourself. All five projects in this course download their data with the same reproducible script pattern.
background: The UCI Machine Learning Repository (1987) was the first big public collection. Kaggle (2010) added competitions and a huge community; OpenML and Hugging Face Datasets made datasets programmatically accessible.
use_cases:
  - Learning and practising on well-known benchmarks (Titanic, MNIST, California housing)
  - Competitions and portfolio projects
  - Pre-training or augmenting data for real projects
where_used:
  - 01-house-prices
  - 02-titanic-survival
  - 03-customer-segmentation
  - 04-spam-detector
  - 05-digit-recognizer
code:
  - mlscratch/download.py:fetch
  - mlscratch/download.py:kaggle_download
demo: demo.py
workflow: |
  flowchart LR
    A[Search Kaggle / UCI / OpenML] --> B[Read the data card: columns, licence, collection]
    B --> C[Kaggle token in ~/.kaggle/kaggle.json]
    C --> D[download_data.py: Kaggle → source → mirror]
    D --> E[Cache in data/raw, never edit]
    E --> F[Record source, version, date]
---

## Where to find data
| Hub | Strengths | Access |
|-----|-----------|--------|
| **Kaggle Datasets & Competitions** | huge variety, notebooks, discussions | Kaggle CLI / API (free account + token) |
| **UCI ML Repository** | classic, well-documented benchmarks | direct download |
| **OpenML** | standardised tasks + metadata | `openml` package / REST API |
| **Hugging Face Datasets** | text, audio, images | `datasets` library |
| **Government open data** (data.gov, data.gov.in, data.europa.eu) | real-world public sector data | CSV / APIs |
| **Google Dataset Search** | search engine over all of the above | web |

## Setting up the Kaggle API (once)
1. Create a free account at kaggle.com → **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`) and on Linux/macOS run `chmod 600 ~/.kaggle/kaggle.json`.
3. `pip install kaggle`, then:
```bash
kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/raw --unzip
kaggle competitions download -c titanic -p data/raw        # accept the competition rules on the website first
```
Or set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` instead of the file.

## Read the data card before modelling
- **Licence** - can you use it commercially? (CC0, CC-BY, research-only...)
- **How was it collected and labelled?** Biases in collection become biases in the model.
- **Column meanings, units and missing-value codes.**
- **Time span** - is it still representative?

## Reproducible downloads
Each project ships a `download_data.py` built on `mlscratch.download.fetch`, which tries sources in order and caches the file:
```python
fetch("titanic.csv", sources=[
    {"kaggle": "c/titanic", "file": "train.csv"},                     # your Kaggle token, if configured
    {"url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"},   # public mirror
], dest_dir="data/raw")
```
Raw files go to `data/raw/` (git-ignored) and are never edited by hand - all cleaning happens in code.

## Exercises
1. Configure your Kaggle token and run any project's `download_data.py`; check which source it used.
2. Pick a dataset on Kaggle and write its data card in your own words (source, licence, rows, columns, target, caveats).
3. Add a SHA-256 checksum to a `fetch` call so a changed file is detected.
