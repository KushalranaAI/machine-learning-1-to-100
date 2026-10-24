"""Step 1 - download the Mall Customers dataset (200 customers: gender, age, annual income, spending score).

The Kaggle dataset "vjchoudhary7/customer-segmentation-tutorial-in-python" is the usual source; the two GitHub copies
are identical apart from the column name "Genre" vs "Gender", which segment.py normalises.
"""

from pathlib import Path

from mlscratch.download import fetch

DATA = Path(__file__).resolve().parent / "data"
path = fetch(
    "Mall_Customers.csv",
    [
        {"kaggle": "vjchoudhary7/customer-segmentation-tutorial-in-python", "file": "Mall_Customers.csv"},
        {"url": "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"},
        {"url": "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"},
    ],
    DATA,
)

import pandas as pd  # noqa: E402

df = pd.read_csv(path)
print(f"\n{len(df)} customers × {df.shape[1]} columns")
print(df.describe(include="all").round(1).to_string())
