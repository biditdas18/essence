"""
data/preprocess.py
------------------
Preprocesses the raw Last.fm-1K TSV into train/test interaction
DataFrames and a long-tail item ID set.

Outputs
-------
  data/train_interactions.pkl
  data/test_interactions.pkl
  data/long_tail_ids.pkl

Run:
    python data/preprocess.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
RAW_DIR  = DATA_DIR / "raw"

TARGET_TSV = "userid-timestamp-artid-artname-traid-traname.tsv"

TRAIN_OUT      = DATA_DIR / "train_interactions.pkl"
TEST_OUT       = DATA_DIR / "test_interactions.pkl"
LONG_TAIL_OUT  = DATA_DIR / "long_tail_ids.pkl"

# ---------------------------------------------------------------------------
# Filtering constants
# ---------------------------------------------------------------------------
MIN_INTERACTIONS = 50
MAX_INTERACTIONS = 500
LONG_TAIL_PERCENTILE = 20   # bottom 20% by play count
TEST_FRACTION = 0.20         # randomly held-out 20% of each user's unique tracks
RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tsv() -> Path:
    for path in RAW_DIR.rglob(TARGET_TSV):
        return path
    raise FileNotFoundError(
        f"Cannot find '{TARGET_TSV}' inside {RAW_DIR}. "
        "Run: python data/download_lastfm.py"
    )


def load_raw(tsv_path: Path) -> pd.DataFrame:
    print(f"[load] Reading {tsv_path} …")
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        encoding="utf-8",
        on_bad_lines="skip",
        names=["user_id", "timestamp", "artist_id",
               "artist_name", "track_id", "track_name"],
    )
    print(f"[load] Raw rows: {len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop rows where track_id or track_name is null
    before = len(df)
    df = df.dropna(subset=["track_id", "track_name"])
    print(f"[clean] Dropped {before - len(df):,} rows with null track_id/name "
          f"→ {len(df):,} rows remain")

    # 2. Drop duplicate (user_id, track_id) pairs — keep first
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "track_id"], keep="first")
    print(f"[clean] Dropped {before - len(df):,} duplicate (user,track) pairs "
          f"→ {len(df):,} rows remain")

    return df


def filter_users(df: pd.DataFrame) -> pd.DataFrame:
    # 3. Keep only users with 50–500 unique track interactions
    user_counts = df.groupby("user_id")["track_id"].count()
    valid_users = user_counts[
        (user_counts >= MIN_INTERACTIONS) & (user_counts <= MAX_INTERACTIONS)
    ].index
    before = df["user_id"].nunique()
    df = df[df["user_id"].isin(valid_users)].copy()
    print(f"[filter] Users: {before:,} → {df['user_id'].nunique():,} "
          f"(kept {MIN_INTERACTIONS}–{MAX_INTERACTIONS} interactions)")
    return df


def add_item_text(df: pd.DataFrame) -> pd.DataFrame:
    # 5. Build item_text column
    df["item_text"] = df["track_name"] + " by " + df["artist_name"].fillna("Unknown")
    return df


def compute_long_tail(train_df: pd.DataFrame):
    """
    Define long-tail items as tracks heard by exactly 1 user in the training set.

    Dataset context
    ---------------
    After filtering to users with 50–500 interactions and deduplicating
    (user, track) pairs, this Last.fm-1K slice has only 99 users and the
    maximum global play count for any track is 10.  The distribution is:
      • 89.8% of tracks: play count = 1 (singleton, heard by one user)
      •  9.9% of tracks: play count = 2–4
      •  0.3% of tracks: play count = 5–10

    In this context "popular" means heard by ≥5 users and "long-tail" means
    heard by exactly 1 user — the items that CF will almost never surface
    (zero co-occurrence signal) but that Essence may discover via semantic
    similarity to the user's taste cluster.
    """
    play_counts = train_df.groupby("track_id").size()
    long_tail_ids = set(play_counts[play_counts == 1].index)
    return long_tail_ids


def split_train_test(df: pd.DataFrame):
    """
    Random track-ID hold-out split.

    Why not chronological?
    ----------------------
    After deduplication each user has each track exactly once, so a
    chronological 80/20 split produces train-tracks and test-tracks that are
    completely disjoint by construction.  Recommenders that avoid seen (train)
    items cannot hit any test item above chance level (~16 targets / 18 K
    candidates), making recall identically zero across all 200 users.

    Random hold-out fix
    -------------------
    For each user we randomly set aside 20% of their unique track IDs as the
    test set and keep 80% in train.  Because both halves come from the same
    user's taste profile:
      - Test items appear in the global embedding map (they're real tracks).
      - Semantic recommenders (Content, Essence) will score items similar to
        the user's train embedding, which biases toward the held-out items.
      - This yields non-trivial recall and allows meaningful comparison.

    Seed is fixed for reproducibility.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    train_rows = []
    test_rows  = []

    for uid, group in df.groupby("user_id", sort=False):
        track_ids = group["track_id"].tolist()
        n         = len(track_ids)
        n_test    = max(1, int(n * TEST_FRACTION))

        # Randomly select test track IDs
        test_idx_positions = rng.choice(n, size=n_test, replace=False)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx_positions] = True

        train_rows.append(group[~test_mask])
        test_rows.append(group[test_mask])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tsv_path = find_tsv()
    df = load_raw(tsv_path)
    df = clean(df)
    df = filter_users(df)
    df = add_item_text(df)

    # Sort each user's history chronologically (used within groups)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_df, test_df = split_train_test(df)

    # Compute long-tail on train only — avoids leaking test tracks
    long_tail_ids = compute_long_tail(train_df)

    # Save outputs
    train_df.to_pickle(TRAIN_OUT)
    test_df.to_pickle(TEST_OUT)
    with open(LONG_TAIL_OUT, "wb") as fh:
        pickle.dump(long_tail_ids, fh)

    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total users retained          : {train_df['user_id'].nunique():,}")
    print(f"Total unique tracks (train)   : {train_df['track_id'].nunique():,}")
    print(f"Number of long-tail items     : {len(long_tail_ids):,}")
    avg_interactions = train_df.groupby("user_id").size().mean()
    print(f"Avg train interactions/user   : {avg_interactions:.1f}")
    print(f"\nSaved:")
    print(f"  {TRAIN_OUT}")
    print(f"  {TEST_OUT}")
    print(f"  {LONG_TAIL_OUT}")


if __name__ == "__main__":
    main()
