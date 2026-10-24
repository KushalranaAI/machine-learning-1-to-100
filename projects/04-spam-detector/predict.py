"""Step 4 - classify your own messages with the saved naive Bayes model (pure Python + NumPy, no training code).

    python predict.py "WINNER!! Claim your £1000 prize, call 09061701461" "are we still on for lunch?"
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mlscratch.text import tokenize

path = Path(__file__).resolve().parent / "data" / "spam_model.json"
if not path.exists():
    raise SystemExit("data/spam_model.json not found - run `python train.py` first.")
m = json.loads(path.read_text())
vocab, flp, prior = m["vocabulary"], np.array(m["feature_log_prob"]), np.array(m["class_log_prior"])

messages = sys.argv[1:] or [
    "WINNER!! You have been selected for a £1000 cash prize. Call 09061701461 now to claim",
    "Hey, are we still on for lunch tomorrow? I can pick you up at 12",
    "URGENT: your mobile number has won 2 free ringtones, text YES to 87121",
    "Sorry, I'll call you later, in a meeting",
    "Free entry to the gym this weekend, see you there?",
]
for text in messages:
    toks = tokenize(text)
    terms = toks + [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
    x = np.zeros(len(vocab))
    for t, c in Counter(terms).items():
        if t in vocab:
            x[vocab[t]] = c
    joint = flp @ x + prior                                  # log P(c) + Σ x_w log θ_c,w
    p_spam = 1 / (1 + np.exp(joint[0] - joint[1]))
    known = [t for t in terms if t in vocab]
    top = sorted(set(known), key=lambda t: flp[0, vocab[t]] - flp[1, vocab[t]])[:3]
    verdict = "SPAM" if p_spam >= m["threshold"] else "ham "
    print(f"{verdict}  P(spam)={p_spam:.3f}  {text[:70]!r}")
    print(f"       most spammy known terms: {', '.join(top) or '-'}")
