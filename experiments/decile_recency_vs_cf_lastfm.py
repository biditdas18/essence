"""
experiments/decile_recency_vs_cf_lastfm.py
------------------------------------------------
Step 5c: Last.fm version -- does Recency-Weighted (and Last-Item,
Avg-Last-10) also beat CF-ItemKNN at true cold-start (decile 1/2), the
same way Essence does? Same decile assignment as decile_analysis_lastfm.py.

Saves:
  results/decile1_peruser_recency_lastfm.csv
  results/decile2_peruser_recency_lastfm.csv

Run:
    python experiments/decile_recency_vs_cf_lastfm.py
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import (
    build_itemknn_model, cf_itemknn_recommend,
    last_item_recommend, avg_last10_recommend, recency_weighted_recommend,
)

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def main():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)

    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))

    item_ids_all = sorted(emb.keys())
    popularity = train_df.groupby("track_id").size()
    pop_series = pd.Series(0, index=item_ids_all, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    itemknn = build_itemknn_model(train_df, item_col="track_id")

    rows_by_decile = {1: [], 2: []}
    systems_fns = {
        "Last-Item": last_item_recommend,
        "Avg-Last-10": avg_last10_recommend,
        "Recency-Weighted": recency_weighted_recommend,
    }

    for uid in tqdm(all_users, desc="Users"):
        actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
        if not actual:
            continue

        recs = {name: fn(uid, train_df, emb, M) for name, fn in systems_fns.items()}
        recs["CF (ItemKNN)"] = cf_itemknn_recommend(uid, train_df, itemknn, M, item_embedding_map=emb)

        test_by_decile = defaultdict(set)
        for iid in actual:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for d in [1, 2]:
            items_in_decile = test_by_decile.get(d)
            if not items_in_decile:
                continue
            items_list = list(items_in_decile)
            for sys_name, rec_list in recs.items():
                r = recall_at_k(rec_list, items_list, k=M)
                rows_by_decile[d].append({"user_id": uid, "system": sys_name,
                                          "recall@10": r, "long_tail_recall@10": ""})

    for d in [1, 2]:
        out_path = RESULTS_DIR / f"decile{d}_peruser_recency_lastfm.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(rows_by_decile[d])
        n = len(set(r["user_id"] for r in rows_by_decile[d]))
        print(f"Saved decile{d}_peruser_recency_lastfm.csv: {len(rows_by_decile[d])} rows ({n} users)")


if __name__ == "__main__":
    main()
