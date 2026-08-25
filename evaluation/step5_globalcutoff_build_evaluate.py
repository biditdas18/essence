"""
evaluation/step5_globalcutoff_build_evaluate.py
--------------------------------------------------
Tier-2 Step 5: rebuild the train/test split using ONE shared global
timestamp cutoff per dataset (the 80th percentile of ALL interaction
timestamps in that dataset's existing sampled population), instead of
the paper's per-user 80/20 chronological split. Reruns the 8-system
evaluation + paired bootstrap + silhouette stratification at each
dataset's existing validated K.

Item embeddings are NOT regenerated: the item universe (all items ever
touched by the existing sampled users, train union test) is unchanged by
moving the train/test boundary -- only which interactions land on which
side of the boundary changes. The existing embeddings_metadata.pkl for
each dataset is reused directly.

Users with zero interactions on one side of the global cutoff (all-train
or all-test) are excluded and the count reported -- a global cutoff
necessarily strands some users this way, unlike the per-user split.

HARD STOP per instructions: this script only builds data and reports
metrics. No paper file is touched by this script or by reading its
output, regardless of what the results show.

Run:
    python evaluation/step5_globalcutoff_build_evaluate.py lastfm
    python evaluation/step5_globalcutoff_build_evaluate.py amazon
    python evaluation/step5_globalcutoff_build_evaluate.py movielens
"""
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "evaluation"))


def global_cutoff_split(df, ts_col="timestamp", pct=0.80):
    """Single global cutoff at the given percentile of ALL timestamps in df.
    Returns (train_df, test_df, n_users_stranded)."""
    cutoff = df[ts_col].quantile(pct)
    train_df = df[df[ts_col] < cutoff].copy()
    test_df = df[df[ts_col] >= cutoff].copy()

    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    all_users = set(df["user_id"].unique())
    both = train_users & test_users
    stranded = all_users - both

    train_df = train_df[train_df["user_id"].isin(both)]
    test_df = test_df[test_df["user_id"].isin(both)]
    print(f"  Global cutoff at {pct:.0%} percentile = {cutoff}")
    print(f"  Users with data on both sides: {len(both):,} / {len(all_users):,} "
          f"({len(stranded):,} stranded: all-train or all-test only, excluded)")
    return train_df, test_df, len(stranded)


def build_lastfm(out_dir: Path):
    print("=" * 60)
    print("[globalcutoff] Last.fm-1K")
    print("=" * 60)
    from data.preprocess import find_tsv, load_raw, clean, filter_users

    tsv_path = find_tsv()
    df = load_raw(tsv_path)
    df = clean(df)
    df = filter_users(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.rename(columns={"track_id": "item_id"})

    train_df, test_df, n_stranded = global_cutoff_split(df)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in counts.items() if c == 1}
    print(f"  Long-tail: {len(longtail):,} of {len(counts):,} train items")

    out_dir.mkdir(exist_ok=True)
    train_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "train.csv", index=False)
    test_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(out_dir / "longtail_items.csv", index=False)
    return n_stranded


def build_amazon(out_dir: Path):
    print("=" * 60)
    print("[globalcutoff] Amazon Books")
    print("=" * 60)
    RAW_DIR = BASE_DIR / "data" / "amazon_raw"
    PROC_DIR = BASE_DIR / "data" / "amazon_processed"

    train0 = pd.read_csv(PROC_DIR / "train.csv")
    test0 = pd.read_csv(PROC_DIR / "test.csv")
    sampled_pairs = pd.concat([train0[["user_id", "item_id"]], test0[["user_id", "item_id"]]])
    valid_pairs = set(zip(sampled_pairs["user_id"], sampled_pairs["item_id"]))
    sampled_users = set(sampled_pairs["user_id"].unique())

    ratings = pd.read_csv(RAW_DIR / "Books_5core_ratings.csv")
    ratings = ratings.rename(columns={"parent_asin": "item_id"})
    ratings = ratings[ratings["user_id"].isin(sampled_users)]
    ratings["pair"] = list(zip(ratings["user_id"], ratings["item_id"]))
    ratings = ratings[ratings["pair"].isin(valid_pairs)].drop(columns=["pair"])
    ratings = ratings.sort_values("timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="first")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"])
    print(f"  Interactions (existing sampled population, all ratings): {len(ratings):,}")

    train_df, test_df, n_stranded = global_cutoff_split(ratings)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in counts.items() if c == 1}
    print(f"  Long-tail: {len(longtail):,} of {len(counts):,} train items")

    out_dir.mkdir(exist_ok=True)
    train_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "train.csv", index=False)
    test_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(out_dir / "longtail_items.csv", index=False)
    return n_stranded


def build_movielens(out_dir: Path):
    print("=" * 60)
    print("[globalcutoff] MovieLens-25M")
    print("=" * 60)
    PROC_DIR = BASE_DIR / "data" / "movielens_processed"
    ratings = pd.read_csv(PROC_DIR / "sampled_ratings.csv")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    print(f"  Interactions (existing sampled population, all ratings): {len(ratings):,}")

    train_df, test_df, n_stranded = global_cutoff_split(ratings)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in counts.items() if c == 1}
    print(f"  Long-tail: {len(longtail):,} of {len(counts):,} train items")

    out_dir.mkdir(exist_ok=True)
    train_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "train.csv", index=False)
    test_df[["user_id", "item_id", "timestamp"]].to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(out_dir / "longtail_items.csv", index=False)
    return n_stranded


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import importlib
    ev = importlib.import_module("step1_ratingthresh_evaluate")

    if which in ("lastfm", "all"):
        out_dir = BASE_DIR / "data" / "lastfm_processed_globalcutoff"
        n_str = build_lastfm(out_dir)
        ev.run_dataset("Last.fm-1K (global cutoff)", out_dir,
                        BASE_DIR / "embeddings" / "item_embeddings.pkl",
                        K=10, out_tag="lastfm_K10", prefix="scratch_globalcutoff")

    if which in ("amazon", "all"):
        out_dir = BASE_DIR / "data" / "amazon_processed_globalcutoff"
        n_str = build_amazon(out_dir)
        ev.run_dataset("Amazon Books (global cutoff)", out_dir,
                        BASE_DIR / "data" / "amazon_processed" / "embeddings_metadata.pkl",
                        K=10, out_tag="amazon_K10", prefix="scratch_globalcutoff")

    if which in ("movielens", "all"):
        out_dir = BASE_DIR / "data" / "movielens_processed_globalcutoff"
        n_str = build_movielens(out_dir)
        ev.run_dataset("MovieLens-25M (global cutoff)", out_dir,
                        BASE_DIR / "data" / "movielens_processed" / "embeddings_metadata.pkl",
                        K=15, out_tag="movielens_K15", prefix="scratch_globalcutoff")


if __name__ == "__main__":
    main()
