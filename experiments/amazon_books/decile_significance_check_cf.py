"""
experiments/amazon_books/decile_significance_check_cf.py
----------------------------------------------------------------
Step 2 (follow-up to Step 7b): the original Amazon decile-1/2 significance
test (decile_significance_check.py) compared Essence against Last-Item,
Avg-Last-10, Recency-Weighted -- NOT CF-ItemKNN, which is what the
MovieLens decile-1 comparison used. This fills that gap so the two
datasets' decile-1 tests are directly comparable.

Same decile assignment as popularity_decile_recall.py (train-set
popularity rank, ties broken by row order, 10 equal-item-count buckets).

Saves per-user CSVs in the schema paired_bootstrap.py expects:
  results/decile1_peruser_cf.csv
  results/decile2_peruser_cf.csv

Run:
    python experiments/amazon_books/decile_significance_check_cf.py
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

from models.recommenders import build_itemknn_model, cf_itemknn_recommend

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
K = 3


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


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


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

        recs_cf = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=42, n_init=10)
                km.fit(np.array(seen_vecs))
                recent_vecs = seen_vecs[-10:]
                if recent_vecs:
                    recent_mean = np.mean(recent_vecs, axis=0)
                    dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                    centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                else:
                    centroid = km.cluster_centers_[0].astype(np.float32)
                recs_essence = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs_essence = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            recs_essence = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for d in [1, 2]:
            items_in_decile = test_by_decile.get(d)
            if not items_in_decile:
                continue
            items_list = list(items_in_decile)
            r_essence = recall_at_k(recs_essence, items_list, k=M)
            r_cf = recall_at_k(recs_cf, items_list, k=M)
            rows_by_decile[d].append({"user_id": uid, "system": "Essence (K=3)",
                                      "recall@10": r_essence, "long_tail_recall@10": ""})
            rows_by_decile[d].append({"user_id": uid, "system": "CF (ItemKNN)",
                                      "recall@10": r_cf, "long_tail_recall@10": ""})

    for d in [1, 2]:
        out_path = RESULTS_DIR / f"decile{d}_peruser_cf.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(rows_by_decile[d])
        n_essence = sum(1 for r in rows_by_decile[d] if r["system"] == "Essence (K=3)")
        print(f"Saved decile{d}_peruser_cf.csv: {len(rows_by_decile[d])} rows ({n_essence} users w/ decile-{d} test items)")


if __name__ == "__main__":
    main()
