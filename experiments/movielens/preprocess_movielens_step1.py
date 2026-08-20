"""
experiments/movielens/preprocess_movielens_step1.py
--------------------------------------------------------
Step 11a.2-4: subsample 2,000 users (same seed/filter convention as
preprocess_amazon.py: seed=42, 20-200 interactions/user), extract the
item set those users actually interacted with, and join links.csv to
get tmdbId for that subset (not the full 62,423-movie catalog).

Outputs:
  data/movielens_processed/sampled_ratings.csv   (userId, movieId, rating, timestamp)
  data/movielens_processed/item_candidates.csv   (movieId, title, genres, tmdbId)

Run:
    python experiments/movielens/preprocess_movielens_step1.py
"""

import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "movielens_raw" / "ml-25m"
PROC_DIR = BASE_DIR / "data" / "movielens_processed"
PROC_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
MIN_INTERACTIONS = 20
MAX_INTERACTIONS = 200
MAX_USERS = 2000


def main():
    print("[1] Loading ratings.csv ...")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    print(f"  {len(ratings):,} ratings, {ratings['userId'].nunique():,} users, {ratings['movieId'].nunique():,} movies")

    print("\n[2] Filtering users to 20-200 interactions (matches preprocess_amazon.py convention) ...")
    user_counts = ratings.groupby("userId").size()
    valid_users = user_counts[(user_counts >= MIN_INTERACTIONS) & (user_counts <= MAX_INTERACTIONS)].index
    print(f"  {len(valid_users):,} users pass the filter")

    print(f"\n[3] Subsampling to {MAX_USERS} users (seed={SEED}) ...")
    rng = random.Random(SEED)
    sampled_uids = rng.sample(sorted(valid_users), MAX_USERS)
    sampled = ratings[ratings["userId"].isin(sampled_uids)].copy()
    print(f"  {len(sampled):,} interactions across {sampled['userId'].nunique():,} users")

    print("\n[4] Extracting item set actually touched by sampled users ...")
    item_ids = sorted(sampled["movieId"].unique())
    print(f"  {len(item_ids):,} unique movies (vs. 62,423 in the full catalog)")

    print("\n[5] Joining movies.csv + links.csv for title/genre/tmdbId ...")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    links = pd.read_csv(RAW_DIR / "links.csv")
    item_meta = movies[movies["movieId"].isin(item_ids)].merge(links, on="movieId", how="left")
    n_missing_tmdb = item_meta["tmdbId"].isna().sum()
    print(f"  {len(item_meta):,} items joined; {n_missing_tmdb} missing tmdbId ({100*n_missing_tmdb/len(item_meta):.1f}%)")

    sampled.to_csv(PROC_DIR / "sampled_ratings.csv", index=False)
    item_meta.to_csv(PROC_DIR / "item_candidates.csv", index=False)
    print(f"\nSaved:\n  {PROC_DIR / 'sampled_ratings.csv'}\n  {PROC_DIR / 'item_candidates.csv'}")


if __name__ == "__main__":
    main()
