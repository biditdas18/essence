"""
data/download_lastfm.py
-----------------------
Downloads and extracts the Last.fm-1K dataset.

Primary source : http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
Fallback source: HuggingFace (mdavolio/lastfm-1k)

Run:
    python data/download_lastfm.py
"""

import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_NAME = "lastfm-dataset-1K.tar.gz"
ARCHIVE_PATH = RAW_DIR / ARCHIVE_NAME
PRIMARY_URL  = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"

TARGET_TSV   = "userid-timestamp-artid-artname-traid-traname.tsv"


def download_primary():
    """Try to download from the MTG mirror."""
    print(f"[download] Attempting primary source:\n  {PRIMARY_URL}")
    response = requests.get(PRIMARY_URL, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    with open(ARCHIVE_PATH, "wb") as fh, tqdm(
        desc=ARCHIVE_NAME,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"[download] Saved to {ARCHIVE_PATH}")


def extract_archive():
    """Extract the tar.gz archive into RAW_DIR."""
    print(f"[extract] Extracting {ARCHIVE_PATH} → {RAW_DIR}")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    print("[extract] Done.")


def fallback_huggingface():
    """Fallback: pull from HuggingFace and write a TSV manually."""
    print("[fallback] Primary source failed. Trying HuggingFace mirror …")
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` not installed. "
            "Run: pip install datasets"
        )

    ds = load_dataset("mdavolio/lastfm-1k", split="train")
    out_path = RAW_DIR / TARGET_TSV
    print(f"[fallback] Writing {len(ds)} rows to {out_path}")

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in tqdm(ds, desc="Writing TSV"):
            line = "\t".join([
                str(row.get("user_id", "")),
                str(row.get("timestamp", "")),
                str(row.get("artist_id", "")),
                str(row.get("artist_name", "")),
                str(row.get("track_id", "")),
                str(row.get("track_name", "")),
            ])
            fh.write(line + "\n")

    print(f"[fallback] TSV saved to {out_path}")


def find_tsv():
    """Search RAW_DIR (recursively) for the target TSV."""
    for path in RAW_DIR.rglob(TARGET_TSV):
        return path
    return None


def main():
    # Skip download if TSV already present
    tsv_path = find_tsv()
    if tsv_path:
        print(f"[skip] Target TSV already exists:\n  {tsv_path}")
        return

    # Try primary download
    try:
        if not ARCHIVE_PATH.exists():
            download_primary()
        extract_archive()
    except Exception as primary_err:
        print(f"[warn] Primary source failed: {primary_err}")
        ARCHIVE_PATH.unlink(missing_ok=True)
        fallback_huggingface()

    # Final check
    tsv_path = find_tsv()
    if tsv_path:
        print(f"\n[ok] Dataset ready at:\n  {tsv_path}")
    else:
        raise FileNotFoundError(
            f"Could not locate '{TARGET_TSV}' inside {RAW_DIR}. "
            "Check download logs above."
        )


if __name__ == "__main__":
    main()
