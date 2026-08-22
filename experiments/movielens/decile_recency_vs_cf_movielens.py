"""
experiments/movielens/decile_recency_vs_cf_movielens.py
--------------------------------------------------------------
Step 5c: MovieLens version -- does Recency-Weighted (and Last-Item,
Avg-Last-10) also beat CF-ItemKNN at true cold-start (decile 1/2), the
same way Essence does? Same decile assignment as decile_analysis_movielens.py.

Saves:
  results/decile1_peruser_recency_movielens.csv
  results/decile2_peruser_recency_movielens.csv

Run:
    python experiments/movielens/decile_recency_vs_cf_movielens.py
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
RECENCY_N = 10
RECENCY_DECAY = 0.9


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k):
    scores = C @ query_vec
    scores = scores.copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


def normalized(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def recency_weighted_query(recent_vecs, decay=RECENCY_DECAY):
    weights = np.array([decay ** i for i in range(len(recent_vecs) - 1, -1, -1)])
    weights = weights / weights.sum()
    return np.average(np.array(recent_vecs), axis=0, weights=weights)


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    itemknn = build_itemknn_model(train_df, item_col="item_id")

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    rows_by_decile = {1: [], 2: []}

    for uid in tqdm(all_users, desc="Users"):
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())
        if not test_items:
            continue

        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        recs = {"CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M)}

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            recs["Last-Item"] = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs["Avg-Last-10"] = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs["Recency-Weighted"] = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)
        else:
            recs["Last-Item"] = recs["Avg-Last-10"] = recs["Recency-Weighted"] = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
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
        out_path = RESULTS_DIR / f"decile{d}_peruser_recency_movielens.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(rows_by_decile[d])
        n = len(set(r["user_id"] for r in rows_by_decile[d]))
        print(f"Saved decile{d}_peruser_recency_movielens.csv: {len(rows_by_decile[d])} rows ({n} users)")


if __name__ == "__main__":
    main()
