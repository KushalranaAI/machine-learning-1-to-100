"""Step 1 - download the California Housing dataset (1990 US census, 20,640 districts).

Sources are tried in order: the Kaggle dataset (needs a token), then the copy used in
"Hands-On Machine Learning" on GitHub. The file is cached in ./data/.
"""

from pathlib import Path

from mlscratch.download import fetch

DATA = Path(__file__).resolve().parent / "data"

path = fetch(
    "housing.csv",
    [
        {"kaggle": "camnugent/california-housing-prices", "file": "housing.csv"},
        {"url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"},
        {"url": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"},
    ],
    DATA,
)

import pandas as pd  # noqa: E402

df = pd.read_csv(path)
print(f"\n{len(df):,} rows × {df.shape[1]} columns")
print(df.head().to_string())
