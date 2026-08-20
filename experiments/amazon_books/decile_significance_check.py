"""
experiments/amazon_books/decile_significance_check.py
------------------------------------------------------------
Step 7b: per-user Recall@10 restricted to popularity decile 1 (rarest
10% of the catalog) and decile 2 (next rarest), for Essence and the
three recency baselines -- needed to run a real paired bootstrap on the
Step 7 finding (Essence trailing at deciles 1/2), which previously only
had aggregate means.

Uses the IDENTICAL decile assignment and recommendation-generation logic
as popularity_decile_recall.py (same KMeans seed=42, same M=10 candidate
scoring) so results are directly comparable/reproducible against that
script's aggregate table.

Saves, in the same schema evaluation/paired_bootstrap.py already reads
(user_id, system, recall@10, long_tail_recall@10 -- the latter left
blank here since this is a single-metric analysis, not a second metric):
  results/decile1_peruser.csv
  results/decile2_peruser.csv

Run:
    python experiments/amazon_books/decile_significance_check.py
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
RECENCY_N = 10
RECENCY_DECAY = 0.9
SYSTEMS = ["Essence (K=3)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k):
    scores = C @ query_vec
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


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    # Identical decile assignment to popularity_decile_recall.py
    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

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

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        recs = {}
        if seen_vecs:
            recs["Last-Item"] = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs["Avg-Last-10"] = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs["Recency-Weighted"] = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)

            if len(seen_vecs) >= 3:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                km.fit(np.array(seen_vecs))
                if recent_vecs:
                    recent_mean = np.mean(recent_vecs, axis=0)
                    dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                    centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                else:
                    centroid = km.cluster_centers_[0].astype(np.float32)
                recs["Essence (K=3)"] = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs["Essence (K=3)"] = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            for s in SYSTEMS:
                recs[s] = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for d in [1, 2]:
            items_in_decile = test_by_decile.get(d)
            if not items_in_decile:
                continue
            for sys_name, rec_list in recs.items():
                hits = len(set(rec_list) & items_in_decile)
                recall = hits / len(items_in_decile)
                rows_by_decile[d].append({
                    "user_id": uid, "system": sys_name,
                    "recall@10": recall, "long_tail_recall@10": "",
                })

    for d in [1, 2]:
        out_path = RESULTS_DIR / f"decile{d}_peruser.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(rows_by_decile[d])
        print(f"Saved {len(rows_by_decile[d])} rows to {out_path}")


if __name__ == "__main__":
    main()
