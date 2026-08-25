"""
evaluation/step2_multiseed_amazon.py
--------------------------------------
Tier-1 Step 2: rerun Amazon Books's full pipeline (resample users with a
new seed -> chronological split -> longtail -> metadata embeddings ->
8-system evaluation -> paired bootstrap) at 2 additional user-sampling
seeds (43, 44), alongside the existing seed=42, to check whether
headline findings are stable across which 2,000 users happened to be
drawn.

Reuses the exact filtering/sampling logic of
experiments/amazon_books/preprocess_amazon.py (20-200 interactions/user,
2,000-user subsample) parametrized by SEED instead of hardcoded 42, and
the vectorized evaluation logic of step1_ratingthresh_evaluate.py.

Outputs (per seed):
  data/amazon_processed_seed{N}/{train,test,longtail_items}.csv
  data/amazon_processed_seed{N}/embeddings_metadata.pkl
  results/scratch_multiseed_amazon_seed{N}_K10.csv
  results/scratch_multiseed_bootstrap_amazon_seed{N}_K10.csv

Run:
    python evaluation/step2_multiseed_amazon.py 43
    python evaluation/step2_multiseed_amazon.py 44
"""
import json
import pickle
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.preprocess import split_train_test

RAW_DIR = BASE_DIR / "data" / "amazon_raw"
MIN_INTERACTIONS = 20
MAX_INTERACTIONS = 200
MAX_USERS = 2000


def preprocess(seed: int, out_dir: Path):
    print(f"[seed={seed}] Preprocessing (resample -> split -> longtail) ...")
    meta = {}
    with open(RAW_DIR / "meta_Books_filtered.jsonl") as f:
        for line in f:
            row = json.loads(line)
            meta[row["item_id"]] = {
                "title": row.get("title", ""), "author": row.get("author", ""),
                "description": row.get("description", ""),
            }

    user_items_raw = defaultdict(list)
    with open(RAW_DIR / "Books_5core_ratings.csv") as f:
        import csv as _csv
        reader = _csv.DictReader(f)
        for row in reader:
            user_items_raw[row["user_id"]].append((row["parent_asin"], row["timestamp"]))

    user_items = {}
    for uid, items in user_items_raw.items():
        seen, deduped = set(), []
        for iid, ts in items:
            if iid not in seen:
                seen.add(iid)
                deduped.append((iid, ts))
        user_items[uid] = deduped

    filtered_users = {uid: items for uid, items in user_items.items()
                       if MIN_INTERACTIONS <= len(items) <= MAX_INTERACTIONS}
    print(f"  Users after 20-200 filter: {len(filtered_users):,}")

    rng = random.Random(seed)
    sampled_uids = rng.sample(sorted(filtered_users.keys()), MAX_USERS) \
        if len(filtered_users) > MAX_USERS else sorted(filtered_users.keys())
    sampled_users = {uid: filtered_users[uid] for uid in sampled_uids}
    print(f"  Users after subsample (seed={seed}): {len(sampled_users):,}")

    records = []
    for uid in sampled_uids:
        for iid, ts in sampled_users[uid]:
            records.append((uid, iid, ts))
    df = pd.DataFrame(records, columns=["user_id", "item_id", "timestamp"])
    train_df, test_df = split_train_test(df)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in counts.items() if c == 1}
    print(f"  Long-tail: {len(longtail):,} of {len(counts):,} train items")

    out_dir.mkdir(exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(out_dir / "longtail_items.csv", index=False)

    unique_items = sorted(set(train_df["item_id"]) | set(test_df["item_id"]))
    return unique_items, meta


def embed(unique_items, meta, out_dir: Path):
    from sentence_transformers import SentenceTransformer
    print(f"  Embedding {len(unique_items):,} items (metadata-only, Pass 1 style) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{meta.get(i, {}).get('title','')} by {meta.get(i, {}).get('author','')}. "
             f"{meta.get(i, {}).get('description','')}" for i in unique_items]
    t0 = time.time()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    print(f"  Embedded in {time.time()-t0:.1f}s")
    emb = {iid: vecs[i] for i, iid in enumerate(unique_items)}
    with open(out_dir / "embeddings_metadata.pkl", "wb") as f:
        pickle.dump(emb, f)
    return emb


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    out_dir = BASE_DIR / "data" / f"amazon_processed_seed{seed}"
    t_total = time.time()

    unique_items, meta = preprocess(seed, out_dir)
    embed(unique_items, meta, out_dir)

    # Reuse the vectorized evaluation logic from step1_ratingthresh_evaluate.py
    sys.path.insert(0, str(BASE_DIR / "evaluation"))
    import importlib
    ev = importlib.import_module("step1_ratingthresh_evaluate")
    ev.run_dataset(
        f"Amazon Books (seed={seed})", out_dir, out_dir / "embeddings_metadata.pkl",
        K=10, out_tag=f"amazon_seed{seed}_K10", prefix="scratch_multiseed",
    )
    print(f"\n[seed={seed}] TOTAL wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
