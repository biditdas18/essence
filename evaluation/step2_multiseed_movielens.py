"""
evaluation/step2_multiseed_movielens.py
------------------------------------------
Tier-1 Step 2: rerun MovieLens-25M's full pipeline (resample users with a
new seed -> item candidates -> TMDb overview fetch (delta only, shared
resumable cache) -> item text -> embeddings -> 8-system evaluation ->
paired bootstrap) at 2 additional user-sampling seeds (43, 44), alongside
the existing seed=42.

TMDb fetch note: experiments/movielens/fetch_tmdb_overviews.py reads its
candidate list from the canonical data/movielens_processed/item_candidates.csv
and writes to the canonical, shared, resumable
data/movielens_processed/tmdb_overviews.jsonl cache (skips already-fetched
tmdbIds). To reuse it without permanently altering the seed=42 canonical
candidates file, this script backs up item_candidates.csv, swaps in the
new seed's candidate list, runs the fetch (only NEW movie IDs get fetched;
already-cached ones are skipped), then restores the original file. The
overview cache itself is only ever added to, never removed from.

Outputs (per seed):
  data/movielens_processed_seed{N}/{train,test,longtail_items}.csv
  data/movielens_processed_seed{N}/embeddings_metadata.pkl
  results/scratch_multiseed_movielens_seed{N}_K15.csv
  results/scratch_multiseed_bootstrap_movielens_seed{N}_K15.csv

Run:
    python evaluation/step2_multiseed_movielens.py 43
    python evaluation/step2_multiseed_movielens.py 44
"""
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.preprocess import split_train_test

RAW_DIR = BASE_DIR / "data" / "movielens_raw" / "ml-25m"
PROC_DIR = BASE_DIR / "data" / "movielens_processed"  # canonical seed=42 dir
MIN_INTERACTIONS = 20
MAX_INTERACTIONS = 200
MAX_USERS = 2000


def sample_and_candidates(seed: int):
    print(f"[seed={seed}] Loading ratings.csv and resampling ...")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    user_counts = ratings.groupby("userId").size()
    valid_users = user_counts[(user_counts >= MIN_INTERACTIONS) & (user_counts <= MAX_INTERACTIONS)].index
    rng = random.Random(seed)
    sampled_uids = rng.sample(sorted(valid_users), MAX_USERS)
    sampled = ratings[ratings["userId"].isin(sampled_uids)].copy()
    print(f"  {len(sampled):,} interactions across {sampled['userId'].nunique():,} users")

    item_ids = sorted(sampled["movieId"].unique())
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    links = pd.read_csv(RAW_DIR / "links.csv")
    item_meta = movies[movies["movieId"].isin(item_ids)].merge(links, on="movieId", how="left")
    n_missing = item_meta["tmdbId"].isna().sum()
    print(f"  {len(item_meta):,} candidate items, {n_missing} missing tmdbId")
    return sampled, item_meta


def fetch_overviews_for_seed(item_meta: pd.DataFrame):
    """Temporarily swap item_candidates.csv, run the resumable fetch
    against the shared cache, then restore the canonical file."""
    canonical_path = PROC_DIR / "item_candidates.csv"
    backup_path = PROC_DIR / "item_candidates.csv.canonical_backup"
    already_backed_up = backup_path.exists()
    if not already_backed_up:
        shutil.copy(canonical_path, backup_path)
    try:
        item_meta.to_csv(canonical_path, index=False)
        print("  Running TMDb fetch (resumable, delta-only against shared cache) ...")
        subprocess.run([sys.executable, "experiments/movielens/fetch_tmdb_overviews.py"],
                        cwd=str(BASE_DIR), check=True)
    finally:
        shutil.copy(backup_path, canonical_path)
        print("  Restored canonical item_candidates.csv")


def build_item_text_for(item_meta: pd.DataFrame):
    import json
    overviews = {}
    with open(PROC_DIR / "tmdb_overviews.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            overviews[int(obj["movieId"])] = obj

    def clean_genres(g):
        if pd.isna(g) or g == "(no genres listed)":
            return "Unknown"
        return g.replace("|", ", ")

    texts = {}
    n_fetched = 0
    for _, row in item_meta.iterrows():
        mid = row["movieId"]
        rec = overviews.get(mid)
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        clean_title = title.split(" (")[0] if title else ""
        genres = clean_genres(row["genres"])
        if rec is not None:
            n_fetched += 1
            overview = (rec.get("overview") or "").strip()
            base = f"{clean_title} — {genres}"
            texts[mid] = f"{base}, {overview}" if overview else base
        else:
            texts[mid] = f"{clean_title} — {genres}"
    print(f"  Item text built for {len(texts):,} items ({n_fetched:,} with a fetched TMDb record)")
    return texts


def embed(texts: dict, out_dir: Path):
    from sentence_transformers import SentenceTransformer
    print(f"  Embedding {len(texts):,} items ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    item_ids = sorted(texts.keys())
    t0 = time.time()
    vecs = model.encode([texts[i] for i in item_ids], batch_size=64,
                         show_progress_bar=False, convert_to_numpy=True)
    print(f"  Embedded in {time.time()-t0:.1f}s")
    emb = {iid: vecs[i] for i, iid in enumerate(item_ids)}
    with open(out_dir / "embeddings_metadata.pkl", "wb") as f:
        pickle.dump(emb, f)
    return emb


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    out_dir = BASE_DIR / "data" / f"movielens_processed_seed{seed}"
    out_dir.mkdir(exist_ok=True)
    t_total = time.time()

    sampled, item_meta = sample_and_candidates(seed)
    fetch_overviews_for_seed(item_meta)
    texts = build_item_text_for(item_meta)

    sampled = sampled.rename(columns={"userId": "user_id", "movieId": "item_id"})
    sampled = sampled.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    train_df, test_df = split_train_test(sampled)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")
    counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in counts.items() if c == 1}
    print(f"  Long-tail: {len(longtail):,} of {len(counts):,} train items")

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(out_dir / "longtail_items.csv", index=False)

    embed(texts, out_dir)

    sys.path.insert(0, str(BASE_DIR / "evaluation"))
    import importlib
    ev = importlib.import_module("step1_ratingthresh_evaluate")
    ev.run_dataset(
        f"MovieLens-25M (seed={seed})", out_dir, out_dir / "embeddings_metadata.pkl",
        K=15, out_tag=f"movielens_seed{seed}_K15", prefix="scratch_multiseed",
    )
    print(f"\n[seed={seed}] TOTAL wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
