"""Step 2 - EDA for text: class balance, message length, and the words that give spam away."""

from collections import Counter

import numpy as np

from common import META_NAMES, load, meta_features
from mlscratch.text import ENGLISH_STOP_WORDS, tokenize
from mlscratch.viz import plt, savefig

df = load()
spam_rate = df.y.mean()
print(f"{len(df):,} unique messages, {spam_rate:.1%} spam.")
print(f"→ 'always ham' already scores {1 - spam_rate:.1%} accuracy, so accuracy is the wrong metric here: use precision/recall/F1.\n")

meta = meta_features(df.text)
print(f"{'feature':<14}{'ham mean':>10}{'spam mean':>11}")
for j, name in enumerate(META_NAMES):
    print(f"{name:<14}{meta[df.y == 0, j].mean():>10.3f}{meta[df.y == 1, j].mean():>11.3f}")

counts = {k: Counter(t for text in df.text[df.y == k] for t in tokenize(text) if t not in ENGLISH_STOP_WORDS) for k in (0, 1)}
total = {k: sum(c.values()) for k, c in counts.items()}
vocab = set(counts[0]) | set(counts[1])
# Smoothed log-odds ratio: log P(word | spam) / P(word | ham) - exactly what multinomial naive Bayes learns.
lor = {w: np.log((counts[1][w] + 1) / (total[1] + len(vocab))) - np.log((counts[0][w] + 1) / (total[0] + len(vocab)))
       for w in vocab if counts[0][w] + counts[1][w] >= 10}
top_spam = sorted(lor, key=lor.get, reverse=True)[:15]
top_ham = sorted(lor, key=lor.get)[:15]
print("\nmost spam-indicative words (log-odds):", ", ".join(f"{w} ({lor[w]:.1f})" for w in top_spam))
print("most ham-indicative words  (log-odds):", ", ".join(f"{w} ({lor[w]:.1f})" for w in top_ham))

P = plt()
fig, axes = P.subplots(1, 3, figsize=(17, 4.3))
lengths = df.text.str.len()
axes[0].hist([lengths[df.y == 0], lengths[df.y == 1]], bins=50, range=(0, 300), label=["ham", "spam"], density=True)
axes[0].set(title="Message length (characters)")
axes[0].legend()
axes[1].barh(top_spam[::-1], [lor[w] for w in top_spam[::-1]], color="#e45756")
axes[1].set(title="Spam words: log P(w|spam)/P(w|ham)")
axes[2].barh(top_ham[::-1], [lor[w] for w in top_ham[::-1]], color="#54a24b")
axes[2].set(title="Ham words")
savefig(fig, "01_text_eda")
