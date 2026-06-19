"""
Phase 1 experiment: chronological 80/20 split on Last.fm-1K.

PURPOSE
-------
Reviewer feedback: Section 3.5 selects the active centroid from the user's
"most recent r interactions", but Section 4.2 uses a random split — the two
contradict because random shuffle destroys temporal ordering.

This script tests whether a chronological split (first 80% of each user's
unique tracks by timestamp = train, last 20% = test) produces non-trivial
recall now that the embedding cache covers train ∪ test (not train-only).

CRITICAL CONTEXT
----------------
The original preprocess.py comment explains why chronological was abandoned:
  "After deduplication each user has each track exactly once, so a
   chronological 80/20 split produces train-tracks and test-tracks that are
   completely disjoint by construction."

This is still true — train and test are ALWAYS disjoint in any split.
The semantic recommenders (Content, Essence) score UNSEEN items by
cosine similarity to the user's taste profile, so what matters is:
  1. Can the profile (from train items) meaningfully point at test items?
  2. Are the test item embeddings available in the cache?

Both conditions hold regardless of split strategy, since embeddings cover
train ∪ test. The question is whether chronological train/test tracks are
MORE or LESS semantically related than random train/test tracks.

HOW THIS SCRIPT WORKS
---------------------
- Loads raw pkl files (train_interactions.pkl, test_interactions.pkl) to
  get the full user universe and all track metadata (item_text, etc.)
- Rebuilds the interaction DataFrame and applies chronological split.
- Reuses the existing item_embeddings.pkl cache (covers train ∪ test).
- Runs all 3 systems and reports Recall@10 and LT-Recall@10.
- Does NOT modify any canonical files.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import cf_recommend, content_recommend, essence_recommend

DATA_DIR       = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


# ─── Chronological split ─────────────────────────────────────────────────────

def chrono_split(df: pd.DataFrame, test_fraction: float = 0.20):
    """
    For each user: sort by timestamp, take first 80% as train, last 20% as test.
    Timestamps in Last.fm-1K are strings like "2009-05-04T23:08:57Z" — sort lexicographically.
    """
    train_rows, test_rows = [], []
    for uid, group in df.groupby("user_id", sort=False):
        group_sorted = group.sort_values("timestamp")
        n       = len(group_sorted)
        n_test  = max(1, round(n * test_fraction))
        n_train = n - n_test
        train_rows.append(group_sorted.iloc[:n_train])
        test_rows.append(group_sorted.iloc[n_train:])
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def compute_long_tail(train_df):
    play_counts   = train_df.groupby("track_id").size()
    long_tail_ids = set(play_counts[play_counts == 1].index)
    return long_tail_ids


# ─── Metrics ─────────────────────────────────────────────────────────────────

def recall_at_k(recs, actual, k=10):
    if not actual: return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def lt_recall_at_k(recs, actual, long_tail_ids, k=10):
    actual_lt = [i for i in actual if i in long_tail_ids]
    if not actual_lt: return None
    return len(set(recs[:k]) & set(actual_lt)) / len(actual_lt)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1 — Chronological split experiment (Last.fm-1K)")
    print("=" * 60)

    # Load canonical pkl files to get the filtered, clean user universe
    print("\n[1] Loading canonical train/test pkls (for user universe) ...")
    canon_train = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    canon_test  = pd.read_pickle(DATA_DIR / "test_interactions.pkl")

    # Reconstruct full DataFrame (train ∪ test) then re-split chronologically
    full_df = pd.concat([canon_train, canon_test]).drop_duplicates(
        subset=["user_id", "track_id"]
    ).reset_index(drop=True)

    print(f"  Full (train∪test) rows: {len(full_df):,}")
    print(f"  Users: {full_df['user_id'].nunique()}")

    # Verify timestamps exist and are non-null
    null_ts = full_df["timestamp"].isna().sum()
    print(f"  Null timestamps: {null_ts}")
    print(f"  Timestamp sample: {full_df['timestamp'].iloc[0]}")

    print("\n[2] Applying chronological 80/20 split ...")
    chrono_train, chrono_test = chrono_split(full_df, test_fraction=0.20)
    print(f"  Chrono train rows: {len(chrono_train):,}")
    print(f"  Chrono test rows : {len(chrono_test):,}")

    # Check timestamp boundaries per user (sanity check)
    sample_uid = full_df["user_id"].iloc[0]
    u_train = chrono_train[chrono_train["user_id"] == sample_uid]["timestamp"]
    u_test  = chrono_test[chrono_test["user_id"] == sample_uid]["timestamp"]
    print(f"\n  [sanity] User {sample_uid[:20]}...")
    print(f"    Train last timestamp : {u_train.max()}")
    print(f"    Test  first timestamp: {u_test.min()}")
    train_end_ok = u_train.max() <= u_test.min()
    print(f"    Temporal ordering OK : {train_end_ok}")

    long_tail_ids = compute_long_tail(chrono_train)
    print(f"\n  Chrono long-tail items (train): {len(long_tail_ids):,}")

    print("\n[3] Loading embedding cache ...")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as fh:
        emb_map = pickle.load(fh)
    print(f"  Embeddings loaded: {len(emb_map):,} items")

    # Coverage check: what fraction of test items have embeddings?
    test_items  = set(chrono_test["track_id"].unique())
    train_items = set(chrono_train["track_id"].unique())
    emb_keys    = set(emb_map.keys())
    test_cov    = len(test_items & emb_keys) / len(test_items) * 100
    train_cov   = len(train_items & emb_keys) / len(train_items) * 100
    print(f"  Train item embedding coverage: {train_cov:.1f}%")
    print(f"  Test  item embedding coverage: {test_cov:.1f}%")

    print("\n[4] Running recommenders on all users (K=3, M=10) ...")
    all_users = sorted(set(chrono_train["user_id"].unique()) &
                       set(chrono_test["user_id"].unique()))

    rows = []
    for uid in tqdm(all_users, desc="Users"):
        actual = chrono_test[chrono_test["user_id"] == uid]["track_id"].tolist()

        systems = {
            "CF (Popularity)":  cf_recommend(uid, chrono_train, M=10),
            "Content (Avg Emb)": content_recommend(uid, chrono_train, emb_map, M=10),
            "Essence (K=3)":    essence_recommend(uid, chrono_train, emb_map, K=3, M=10),
        }

        for name, recs in systems.items():
            rows.append({
                "user_id": uid,
                "system":  name,
                "recall@10": recall_at_k(recs, actual),
                "lt_recall@10": lt_recall_at_k(recs, actual, long_tail_ids),
            })

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("RESULTS — Chronological split")
    print("=" * 60)
    for sys_name in ["CF (Popularity)", "Content (Avg Emb)", "Essence (K=3)"]:
        sub    = results_df[results_df["system"] == sys_name]
        r10    = sub["recall@10"].mean()
        lt_sub = sub["lt_recall@10"].dropna()
        lt_r10 = lt_sub.mean() if len(lt_sub) > 0 else float("nan")
        lt_n   = len(lt_sub)
        print(f"  {sys_name:<22}  Recall@10={r10:.4f}  LT-Recall@10={lt_r10:.4f}  (LT users={lt_n})")

    print("\n" + "=" * 60)
    print("COMPARISON — Random split (v2 reference) vs Chronological")
    print("=" * 60)
    ref = {
        "CF (Popularity)":   (0.0081, 0.0000),
        "Content (Avg Emb)": (0.0414, 0.0032),
        "Essence (K=3)":     (0.0616, 0.0146),
    }
    print(f"  {'System':<22}  {'R@10 rnd':>9}  {'R@10 chr':>9}  {'LT rnd':>8}  {'LT chr':>8}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")
    for sys_name in ["CF (Popularity)", "Content (Avg Emb)", "Essence (K=3)"]:
        sub    = results_df[results_df["system"] == sys_name]
        r_chr  = sub["recall@10"].mean()
        lt_chr = sub["lt_recall@10"].dropna().mean()
        r_rnd, lt_rnd = ref[sys_name]
        print(f"  {sys_name:<22}  {r_rnd:>9.4f}  {r_chr:>9.4f}  {lt_rnd:>8.4f}  {lt_chr:>8.4f}")

    print("\n[Phase 1 complete — stopping for review]")


if __name__ == "__main__":
    main()
