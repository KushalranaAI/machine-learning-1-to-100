"""Robust dataset download: try Kaggle, then the original source, then public mirrors - and cache the result.

Every project's `download_data.py` uses `fetch()`, so learners get the data however their network is set up.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path


def _http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    """GET with retries and exponential back-off (1s, 2s, 4s) - the polite way to call any server."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ml-1-to-100-course/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network errors, HTTP errors, timeouts
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def kaggle_available() -> bool:
    """True when the Kaggle CLI is installed and an API token is configured."""
    has_token = (Path.home() / ".kaggle" / "kaggle.json").exists() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
    return has_token and shutil.which("kaggle") is not None


def kaggle_download(ref: str, dest: Path, competition: bool = False) -> bool:
    """`kaggle datasets download -d owner/name` (or `competitions download -c name`) into dest, unzipped."""
    if not kaggle_available():
        return False
    cmd = ["kaggle", "competitions", "download", "-c", ref] if competition else ["kaggle", "datasets", "download", "-d", ref]
    try:
        subprocess.run(cmd + ["-p", str(dest), "--force"], check=True, capture_output=True, timeout=600)
    except Exception as exc:
        print(f"  kaggle download failed: {exc}")
        return False
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    return True


def fetch(filename: str, sources: list[dict], dest_dir: str | Path, sha256: str | None = None) -> Path:
    """Make sure `dest_dir/filename` exists, trying each source in order.

    A source is one of:
        {"kaggle": "owner/dataset", "file": "name-in-archive.csv"}            (needs a Kaggle token)
        {"url": "https://.../file.csv"}                                        (direct file)
        {"url": "https://.../archive.zip", "member": "path/in/zip.txt"}        (file inside a zip)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists():
        print(f"✓ {target} already downloaded")
        return target
    for src in sources:
        label = src.get("kaggle") or src.get("url")
        print(f"→ trying {label}")
        try:
            if "kaggle" in src:
                tmp = dest_dir / "_kaggle"
                if kaggle_download(src["kaggle"], tmp, src.get("competition", False)):
                    found = next(tmp.rglob(src["file"]), None)
                    if found:
                        shutil.move(str(found), target)
                        shutil.rmtree(tmp, ignore_errors=True)
                        break
                shutil.rmtree(tmp, ignore_errors=True)
                print("  (skipped: no Kaggle token/CLI - see curriculum/01-data-collection/05-public-datasets-kaggle)")
                continue
            data = _http_get(src["url"])
            if "member" in src:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    data = zf.read(src["member"])
            target.write_bytes(data)
            break
        except Exception as exc:
            print(f"  failed: {exc}")
    else:
        raise RuntimeError(f"Could not download {filename} from any source. Download it manually into {dest_dir}.")
    if sha256 and hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
        target.unlink()
        raise RuntimeError(f"checksum mismatch for {filename}")
    print(f"✓ saved {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return target
