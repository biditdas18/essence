"""
experiments/mind_comirec/dataset.py
--------------------------------------
Loads Last.fm-1K or Amazon Books into a common in-memory format for the
MIND / ComiRec implementations, reusing the EXACT existing temporal
train/test split and item-embedding caches already used by Essence and
the other baselines (does not regenerate any split).

Common format
-------------
item_embedding_map : {item_id: np.ndarray(384,)}   (train ∪ test universe;
                      same cache Essence/Content already score against)
train_sequences     : {user_id: [item_id, ...]}     (chronological, train only)
test_items          : {user_id: [item_id, ...]}
long_tail_ids       : set(item_id)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


def load_lastfm():
    data_dir = BASE_DIR / "data"
    emb_dir = BASE_DIR / "embeddings"

    train_df = pd.read_pickle(data_dir / "train_interactions.pkl")
    test_df = pd.read_pickle(data_dir / "test_interactions.pkl")
    with open(emb_dir / "item_embeddings.pkl", "rb") as f:
        item_embedding_map = pickle.load(f)
    with open(data_dir / "long_tail_ids.pkl", "rb") as f:
        long_tail_ids = pickle.load(f)

    train_sequences = {}
    for uid, g in train_df.sort_values("timestamp").groupby("user_id", sort=False):
        train_sequences[uid] = list(g["track_id"])

    test_items = {}
    for uid, g in test_df.groupby("user_id", sort=False):
        test_items[uid] = list(g["track_id"])

    return item_embedding_map, train_sequences, test_items, long_tail_ids


def load_amazon():
    proc_dir = BASE_DIR / "data" / "amazon_processed"

    train_df = pd.read_csv(proc_dir / "train.csv")
    test_df = pd.read_csv(proc_dir / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index  # row order = chronological (Phase-3 sort)

    with open(proc_dir / "embeddings_metadata.pkl", "rb") as f:
        item_embedding_map = pickle.load(f)
    lt_df = pd.read_csv(proc_dir / "longtail_items.csv")
    long_tail_ids = set(lt_df["item_id"])

    train_sequences = {}
    for uid, g in train_df.sort_values("timestamp").groupby("user_id", sort=False):
        train_sequences[uid] = list(g["item_id"])

    test_items = {}
    for uid, g in test_df.groupby("user_id", sort=False):
        test_items[uid] = list(g["item_id"])

    return item_embedding_map, train_sequences, test_items, long_tail_ids


def load_movielens():
    proc_dir = BASE_DIR / "data" / "movielens_processed"

    train_df = pd.read_csv(proc_dir / "train.csv")
    test_df = pd.read_csv(proc_dir / "test.csv")

    with open(proc_dir / "embeddings_metadata.pkl", "rb") as f:
        item_embedding_map = pickle.load(f)
    lt_df = pd.read_csv(proc_dir / "longtail_items.csv")
    long_tail_ids = set(lt_df["item_id"])

    train_sequences = {}
    for uid, g in train_df.sort_values("timestamp").groupby("user_id", sort=False):
        train_sequences[uid] = list(g["item_id"])

    test_items = {}
    for uid, g in test_df.groupby("user_id", sort=False):
        test_items[uid] = list(g["item_id"])

    return item_embedding_map, train_sequences, test_items, long_tail_ids


LOADERS = {"lastfm": load_lastfm, "amazon": load_amazon, "movielens": load_movielens}


def load_dataset(name: str):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {list(LOADERS)}")
    return LOADERS[name]()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "lastfm"
    item_embedding_map, train_sequences, test_items, long_tail_ids = load_dataset(name)
    n_train_interactions = sum(len(v) for v in train_sequences.values())
    n_test_interactions = sum(len(v) for v in test_items.values())
    print(f"[{name}] users(train)={len(train_sequences)} users(test)={len(test_items)} "
          f"items(emb)={len(item_embedding_map)} long_tail={len(long_tail_ids)}")
    print(f"[{name}] train interactions={n_train_interactions} test interactions={n_test_interactions}")
