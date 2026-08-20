"""
experiments/movielens/preprocess_movielens_step2.py
--------------------------------------------------------
Step 11c: chronological split + long-tail computation for MovieLens,
reusing data/preprocess.py's split_train_test() function directly (it
only touches user_id/timestamp columns, so it's fully reusable without
modification -- not reimplemented).

Long-tail definition matches the Amazon/Last.fm convention: items with
exactly 1 train interaction (singleton), computed on train only.

Outputs:
  data/movielens_processed/train.csv
  data/movielens_processed/test.csv
  data/movielens_processed/longtail_items.csv

Run:
    python experiments/movielens/preprocess_movielens_step2.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from data.preprocess import split_train_test  # reused, not reimplemented

PROC_DIR = BASE_DIR / "data" / "movielens_processed"


def main():
    ratings = pd.read_csv(PROC_DIR / "sampled_ratings.csv")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    print(f"[preprocess2] {len(ratings):,} interactions, {ratings['user_id'].nunique():,} users, "
          f"{ratings['item_id'].nunique():,} items")

    train_df, test_df = split_train_test(ratings)
    print(f"[preprocess2] Train: {len(train_df):,} interactions  Test: {len(test_df):,} interactions")

    popularity = train_df.groupby("item_id").size()
    longtail_ids = set(popularity[popularity == 1].index)
    print(f"[preprocess2] Long-tail (singleton) items: {len(longtail_ids):,} "
          f"({100*len(longtail_ids)/train_df['item_id'].nunique():.1f}% of train catalog)")

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    test_df.to_csv(PROC_DIR / "test.csv", index=False)
    pd.DataFrame({"item_id": sorted(longtail_ids)}).to_csv(PROC_DIR / "longtail_items.csv", index=False)
    print(f"[preprocess2] Saved train.csv, test.csv, longtail_items.csv to {PROC_DIR}")


if __name__ == "__main__":
    main()
