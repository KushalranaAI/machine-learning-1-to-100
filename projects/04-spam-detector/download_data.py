"""Step 1 - download the SMS Spam Collection (5,574 real SMS messages labelled ham/spam; Almeida & Hidalgo, 2011).

Sources: Kaggle (uciml/sms-spam-collection-dataset, a latin-1 CSV with columns v1/v2), the UCI repository zip
(tab-separated), and a GitHub mirror of the UCI file. common.load() understands both formats.
"""

from pathlib import Path

from mlscratch.download import fetch

DATA = Path(__file__).resolve().parent / "data"
path = fetch(
    "sms_spam.raw",
    [
        {"kaggle": "uciml/sms-spam-collection-dataset", "file": "spam.csv"},
        {"url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", "member": "SMSSpamCollection"},
        {"url": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"},
    ],
    DATA,
)

from common import load  # noqa: E402

df = load()
print(f"\n{len(df):,} messages: {df.label.value_counts().to_dict()}")
for label in ("ham", "spam"):
    print(f"\n{label} examples:")
    for text in df[df.label == label].text.head(3):
        print("  •", text[:110])
