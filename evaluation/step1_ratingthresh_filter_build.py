"""
evaluation/step1_ratingthresh_filter_build.py
----------------------------------------------
Tier-1 Step 1: build rating>=4 filtered train/test/longtail data for
Amazon Books and MovieLens-25M, matching each dataset's own rating scale
(checked directly, not assumed: Amazon is 1-5 integer stars, MovieLens-25M
is 0.5-5.0 half-star). Reuses the existing sampled user population (seed
42) and existing item embeddings (embeddings are per-item metadata text,
unaffected by which users rated an item positively) -- only the
interaction set is filtered.

After filtering to rating>=4, the same MIN_INTERACTIONS=20 floor is
re-applied (a user who drops below 20 positive interactions is dropped
entirely, consistent with the original preprocessing's inclusion
criterion) and reported.

Outputs (separate directory, does not touch existing data/*_processed/):
  data/amazon_processed_rt4/{train,test,longtail_items}.csv
  data/movielens_processed_rt4/{train,test,longtail_items}.csv

Run:
    python evaluation/step1_ratingthresh_filter_build.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.preprocess import split_train_test  # reused, not reimplemented

RATING_THRESHOLD = 4.0
MIN_INTERACTIONS = 20


def chrono_split(df, uid_col="user_id"):
    """Chronological 80/20 split per user (matches data/preprocess.py logic,
    generalized to Amazon's user_id/item_id column names)."""
    df2 = df.rename(columns={uid_col: "user_id"}) if uid_col != "user_id" else df
    return split_train_test(df2)


def build_amazon():
    print("=" * 60)
    print("[ratingthresh] Amazon Books")
    print("=" * 60)
    RAW_DIR = BASE_DIR / "data" / "amazon_raw"
    PROC_DIR = BASE_DIR / "data" / "amazon_processed"
    OUT_DIR = BASE_DIR / "data" / "amazon_processed_rt4"
    OUT_DIR.mkdir(exist_ok=True)

    ratings = pd.read_csv(RAW_DIR / "Books_5core_ratings.csv")
    print(f"  Raw ratings: {len(ratings):,}  scale check: min={ratings['rating'].min()}, max={ratings['rating'].max()}")
    ratings = ratings.rename(columns={"parent_asin": "item_id"})

    # Existing sampled population: union of train.csv/test.csv (user_id, item_id) pairs
    train0 = pd.read_csv(PROC_DIR / "train.csv")
    test0 = pd.read_csv(PROC_DIR / "test.csv")
    sampled_pairs = pd.concat([train0[["user_id", "item_id"]], test0[["user_id", "item_id"]]])
    sampled_users = set(sampled_pairs["user_id"].unique())
    print(f"  Existing sampled users: {len(sampled_users):,}")

    # Restrict raw ratings to the existing sampled user population + item pairs already in use
    valid_pairs = set(zip(sampled_pairs["user_id"], sampled_pairs["item_id"]))
    ratings = ratings[ratings["user_id"].isin(sampled_users)]
    ratings["pair"] = list(zip(ratings["user_id"], ratings["item_id"]))
    ratings = ratings[ratings["pair"].isin(valid_pairs)].drop(columns=["pair"])
    # Dedup (user_id, item_id) keep first, consistent with original preprocessing
    ratings = ratings.sort_values("timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="first")
    print(f"  Ratings matched to sampled (user,item) pairs: {len(ratings):,}")

    pos = ratings[ratings["rating"] >= RATING_THRESHOLD].copy()
    print(f"  Positive (rating>={RATING_THRESHOLD}) interactions: {len(pos):,} "
          f"({100*len(pos)/len(ratings):.1f}% of matched interactions)")

    counts = pos.groupby("user_id").size()
    keep_users = counts[counts >= MIN_INTERACTIONS].index
    dropped = len(sampled_users) - len(keep_users)
    print(f"  Users with >= {MIN_INTERACTIONS} positive interactions: {len(keep_users):,} "
          f"(dropped {dropped:,} of {len(sampled_users):,} for falling below the floor)")
    pos = pos[pos["user_id"].isin(keep_users)]

    train_df, test_df = chrono_split(pos)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    train_item_counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in train_item_counts.items() if c == 1}
    print(f"  Long-tail (train singletons): {len(longtail):,} of {len(train_item_counts):,} train items")

    train_df[["user_id", "item_id", "timestamp"]].to_csv(OUT_DIR / "train.csv", index=False)
    test_df[["user_id", "item_id", "timestamp"]].to_csv(OUT_DIR / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(OUT_DIR / "longtail_items.csv", index=False)
    print(f"  Saved to {OUT_DIR}")
    return len(keep_users), dropped


def build_movielens():
    print("=" * 60)
    print("[ratingthresh] MovieLens-25M")
    print("=" * 60)
    PROC_DIR = BASE_DIR / "data" / "movielens_processed"
    OUT_DIR = BASE_DIR / "data" / "movielens_processed_rt4"
    OUT_DIR.mkdir(exist_ok=True)

    ratings = pd.read_csv(PROC_DIR / "sampled_ratings.csv")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    print(f"  Raw sampled ratings: {len(ratings):,}  scale check: "
          f"min={ratings['rating'].min()}, max={ratings['rating'].max()}, "
          f"unique values={sorted(ratings['rating'].unique())}")

    pos = ratings[ratings["rating"] >= RATING_THRESHOLD].copy()
    print(f"  Positive (rating>={RATING_THRESHOLD}) interactions: {len(pos):,} "
          f"({100*len(pos)/len(ratings):.1f}% of all interactions)")

    n_users_before = ratings["user_id"].nunique()
    counts = pos.groupby("user_id").size()
    keep_users = counts[counts >= MIN_INTERACTIONS].index
    dropped = n_users_before - len(keep_users)
    print(f"  Users with >= {MIN_INTERACTIONS} positive interactions: {len(keep_users):,} "
          f"(dropped {dropped:,} of {n_users_before:,} for falling below the floor)")
    pos = pos[pos["user_id"].isin(keep_users)]

    train_df, test_df = split_train_test(pos)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    train_item_counts = Counter(train_df["item_id"])
    longtail = {iid for iid, c in train_item_counts.items() if c == 1}
    print(f"  Long-tail (train singletons): {len(longtail):,} of {len(train_item_counts):,} train items")

    train_df[["user_id", "item_id", "timestamp"]].to_csv(OUT_DIR / "train.csv", index=False)
    test_df[["user_id", "item_id", "timestamp"]].to_csv(OUT_DIR / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail)}).to_csv(OUT_DIR / "longtail_items.csv", index=False)
    print(f"  Saved to {OUT_DIR}")
    return len(keep_users), dropped


if __name__ == "__main__":
    a_keep, a_drop = build_amazon()
    m_keep, m_drop = build_movielens()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Amazon Books:   {a_keep} users retained, {a_drop} dropped (< {MIN_INTERACTIONS} positive interactions)")
    print(f"MovieLens-25M:  {m_keep} users retained, {m_drop} dropped (< {MIN_INTERACTIONS} positive interactions)")
