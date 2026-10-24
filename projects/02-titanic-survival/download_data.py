"""Step 1 - download the Titanic passenger list (891 passengers with known outcome).

Kaggle's "Titanic - Machine Learning from Disaster" competition is the canonical source. You need a token, and you
must click "I understand and accept" on the competition's rules page once. The competition also gives an unlabelled
test.csv for leaderboard submissions. Without Kaggle we fall back to public copies of the same train.csv.
"""

from pathlib import Path

from mlscratch.download import fetch

DATA = Path(__file__).resolve().parent / "data"

path = fetch(
    "train.csv",
    [
        {"kaggle": "titanic", "competition": True, "file": "train.csv"},
        {"url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"},
        {"url": "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"},
    ],
    DATA,
)
if not (DATA / "test.csv").exists():
    print("(no Kaggle test.csv - fine: train.py holds out 20% of train.csv as its own test set)")

import pandas as pd  # noqa: E402

df = pd.read_csv(path)
print(f"\n{len(df)} passengers × {df.shape[1]} columns")
print(df.head().to_string())
