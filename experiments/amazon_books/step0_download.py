"""
Step 0 — Download Amazon Reviews 2023 Books 5-core dataset.

Downloads:
  1. 5-core ratings CSV (user_id, parent_asin, rating, timestamp)
  2. Streams full reviews file → filters to 5-core user/item pairs → saves review text
  3. Streams metadata file → filters to 5-core items → saves item metadata

Output:
  data/amazon_raw/Books_5core_ratings.csv      (interactions)
  data/amazon_raw/Books_reviews_filtered.jsonl (review text per user-item)
  data/amazon_raw/meta_Books_filtered.jsonl    (item metadata)
"""

import gzip
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_URL  = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz"
REVIEWS_URL  = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz"
METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"

RATINGS_OUT  = RAW_DIR / "Books_5core_ratings.csv"
REVIEWS_OUT  = RAW_DIR / "Books_reviews_filtered.jsonl"
META_OUT     = RAW_DIR / "meta_Books_filtered.jsonl"


def download_full(url: str, dest: Path, desc: str) -> None:
    """Download compressed file, decompress, and save."""
    print(f"\n[{desc}] Downloading {url}")
    t0 = time.time()
    req = urllib.request.urlopen(url, timeout=120)
    compressed = req.read()
    elapsed = time.time() - t0
    print(f"  Downloaded {len(compressed)/1e6:.1f} MB in {elapsed:.1f}s")

    with gzip.open(io.BytesIO(compressed), "rb") as gz:
        raw = gz.read()
    dest.write_bytes(raw)
    print(f"  Saved {len(raw)/1e6:.1f} MB to {dest}")


def stream_filter_reviews(url: str, dest: Path, valid_pairs: set) -> int:
    """Stream reviews file; write only rows where (user_id, parent_asin) in valid_pairs."""
    print(f"\n[Reviews] Streaming {url}")
    print(f"  Filtering to {len(valid_pairs):,} 5-core (user, item) pairs ...")
    t0 = time.time()
    req = urllib.request.urlopen(url, timeout=120)

    kept = 0
    read_bytes = 0
    buf = b""
    CHUNK = 4 * 1024 * 1024  # 4 MB chunks

    with open(dest, "w") as out, gzip.open(req, "rt") as gz:
        for i, line in enumerate(gz):
            if i % 500_000 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"  ... scanned {i:,} rows, kept {kept:,} | {elapsed:.0f}s elapsed")
            try:
                row = json.loads(line)
            except Exception:
                continue
            uid  = row.get("user_id", "")
            asin = row.get("parent_asin", "")
            if (uid, asin) in valid_pairs:
                out.write(json.dumps({
                    "user_id":    uid,
                    "item_id":    asin,
                    "review_text": row.get("text", ""),
                    "timestamp":  row.get("timestamp", ""),
                }) + "\n")
                kept += 1

    elapsed = time.time() - t0
    print(f"  Done: kept {kept:,} reviews in {elapsed:.0f}s → {dest}")
    return kept


def stream_filter_meta(url: str, dest: Path, valid_items: set) -> int:
    """Stream metadata file; write only rows where parent_asin in valid_items."""
    print(f"\n[Metadata] Streaming {url}")
    print(f"  Filtering to {len(valid_items):,} 5-core items ...")
    t0 = time.time()
    req = urllib.request.urlopen(url, timeout=120)

    kept = 0
    with open(dest, "w") as out, gzip.open(req, "rt") as gz:
        for i, line in enumerate(gz):
            if i % 500_000 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"  ... scanned {i:,} items, kept {kept:,} | {elapsed:.0f}s elapsed")
            try:
                row = json.loads(line)
            except Exception:
                continue
            asin = row.get("parent_asin", "")
            if asin in valid_items:
                # Extract author name from dict
                author_raw = row.get("author", "")
                if isinstance(author_raw, dict):
                    author = author_raw.get("name", "")
                elif isinstance(author_raw, str) and author_raw.startswith("{"):
                    try:
                        author = eval(author_raw).get("name", "")  # noqa: S307
                    except Exception:
                        author = author_raw[:50]
                else:
                    author = str(author_raw)[:100]

                # Extract description (list → first element)
                desc_raw = row.get("description", "")
                if isinstance(desc_raw, list):
                    description = " ".join(str(x) for x in desc_raw)[:200]
                elif isinstance(desc_raw, str) and desc_raw.startswith("["):
                    try:
                        parsed = eval(desc_raw)  # noqa: S307
                        description = " ".join(str(x) for x in parsed)[:200]
                    except Exception:
                        description = desc_raw[:200]
                else:
                    description = str(desc_raw)[:200]

                out.write(json.dumps({
                    "item_id":     asin,
                    "title":       row.get("title", ""),
                    "author":      author,
                    "description": description,
                }) + "\n")
                kept += 1

    elapsed = time.time() - t0
    print(f"  Done: kept {kept:,} items in {elapsed:.0f}s → {dest}")
    return kept


def main():
    # ─── Step 1: Download ratings CSV ─────────────────────────────────────────
    if RATINGS_OUT.exists():
        print(f"[Ratings] Already exists: {RATINGS_OUT} ({RATINGS_OUT.stat().st_size/1e6:.1f} MB)")
    else:
        download_full(RATINGS_URL, RATINGS_OUT, "Ratings")

    # ─── Load ratings to get 5-core user/item universe ────────────────────────
    print("\n[Ratings] Loading user/item universe ...")
    import csv
    pairs    = set()
    users    = set()
    items    = set()
    with open(RATINGS_OUT) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid  = row["user_id"]
            asin = row["parent_asin"]
            pairs.add((uid, asin))
            users.add(uid)
            items.add(asin)

    print(f"  5-core universe: {len(users):,} users, {len(items):,} items, {len(pairs):,} interactions")

    # ─── Step 2: Stream reviews ────────────────────────────────────────────────
    if REVIEWS_OUT.exists():
        print(f"\n[Reviews] Already exists: {REVIEWS_OUT} ({REVIEWS_OUT.stat().st_size/1e6:.1f} MB)")
    else:
        stream_filter_reviews(REVIEWS_URL, REVIEWS_OUT, pairs)

    # ─── Step 3: Stream metadata ───────────────────────────────────────────────
    if META_OUT.exists():
        print(f"\n[Metadata] Already exists: {META_OUT} ({META_OUT.stat().st_size/1e6:.1f} MB)")
    else:
        stream_filter_meta(METADATA_URL, META_OUT, items)

    # ─── CHECKPOINT 0 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECKPOINT 0 — File summary")
    print("="*60)

    for path, label in [(RATINGS_OUT, "Ratings CSV"), (REVIEWS_OUT, "Reviews JSONL"), (META_OUT, "Metadata JSONL")]:
        size_mb = path.stat().st_size / 1e6
        print(f"\n{label}: {path.name}  ({size_mb:.2f} MB)")

    print("\n--- First 3 rows: Ratings ---")
    with open(RATINGS_OUT) as f:
        for i, line in enumerate(f):
            print(f"  {line.rstrip()}")
            if i >= 3: break

    print("\n--- First 3 rows: Reviews ---")
    with open(REVIEWS_OUT) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            print(f"  {json.dumps({k: str(v)[:80] for k, v in row.items()})}")
            if i >= 2: break

    print("\n--- First 3 rows: Metadata ---")
    with open(META_OUT) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            print(f"  {json.dumps({k: str(v)[:80] for k, v in row.items()})}")
            if i >= 2: break

    print("\n" + "="*60)
    print("CHECKPOINT 0 COMPLETE — waiting for confirmation")
    print("="*60)


if __name__ == "__main__":
    main()
