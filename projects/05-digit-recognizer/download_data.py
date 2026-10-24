"""Step 1 - download MNIST (70,000 handwritten 28×28 digits: 60k train + 10k test) into data/mnist.npz.

Tried in order:
  1. the original IDX files (LeCun et al.) from the public mirror used by torchvision
  2. Keras's single-file mnist.npz
  3. Kaggle's "digit-recognizer" competition (42,000 labelled rows; we keep the last 7,000 as a test set)
Formats are converted to one npz with x_train (uint8, n×28×28), y_train, x_test, y_test.
"""

import gzip
import io
from pathlib import Path

import numpy as np

from mlscratch.download import _http_get, kaggle_download

DATA = Path(__file__).resolve().parent / "data"
TARGET = DATA / "mnist.npz"


def parse_idx(raw: bytes) -> np.ndarray:
    """IDX format: 2 zero bytes, a type code, the number of dims, then big-endian int32 sizes, then the data."""
    ndim = raw[3]
    shape = np.frombuffer(raw, ">i4", count=ndim, offset=4)
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def from_idx():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    parts = {}
    for key, name in [("x_train", "train-images-idx3-ubyte"), ("y_train", "train-labels-idx1-ubyte"),
                      ("x_test", "t10k-images-idx3-ubyte"), ("y_test", "t10k-labels-idx1-ubyte")]:
        print(f"→ {base}{name}.gz")
        parts[key] = parse_idx(gzip.decompress(_http_get(base + name + ".gz")))
    return parts


def from_keras():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print(f"→ {url}")
    with np.load(io.BytesIO(_http_get(url))) as z:
        return {k: z[k] for k in ("x_train", "y_train", "x_test", "y_test")}


def from_kaggle():
    import pandas as pd

    tmp = DATA / "_kaggle"
    print("→ kaggle competitions download -c digit-recognizer")
    if not kaggle_download("digit-recognizer", tmp, competition=True):
        raise RuntimeError("no Kaggle token/CLI")
    df = pd.read_csv(tmp / "train.csv")
    y, x = df.pop("label").to_numpy(np.uint8), df.to_numpy(np.uint8).reshape(-1, 28, 28)
    return {"x_train": x[:-7000], "y_train": y[:-7000], "x_test": x[-7000:], "y_test": y[-7000:]}


if TARGET.exists():
    print(f"✓ {TARGET} already downloaded")
else:
    DATA.mkdir(parents=True, exist_ok=True)
    for source in (from_idx, from_keras, from_kaggle):
        try:
            arrays = source()
            break
        except Exception as exc:
            print(f"  failed: {exc}")
    else:
        raise SystemExit("Could not download MNIST from any source.")
    np.savez_compressed(TARGET, **arrays)
    print(f"✓ saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")

with np.load(TARGET) as z:
    for k in z.files:
        print(f"  {k:<8} {z[k].shape} {z[k].dtype}")
    print("  train label counts:", np.bincount(z["y_train"]).tolist())
