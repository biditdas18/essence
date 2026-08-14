"""
experiments/amazon_books/multi_cutoff_amazon.py
----------------------------------------------------
Step 6: Recall@k / LT-Recall@k at k=5, 10, 20 for Essence and the three
recency baselines, Amazon Books (Pass 1, metadata embeddings). Retrieval
run once at M=20, sliced three ways (not rerun per cutoff).

Run:
    python experiments/amazon_books/multi_cutoff_amazon.py
"""

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

from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 20
CUTOFFS = [5, 10, 20]
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
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())
    n_items = len(item_ids)

    per_system_cutoff_recall = {s: {k: [] for k in CUTOFFS} for s in
                                ["Essence (K=3)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]}
    per_system_cutoff_lt = {s: {k: [] for k in CUTOFFS} for s in
                            ["Essence (K=3)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]}

    for uid in tqdm(all_users, desc="Users"):
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())

        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]

        recs = {}
        if seen_vecs:
            user_vec = normalized(np.mean(seen_vecs, axis=0))
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
                centroid = normalized(centroid)
                recs["Essence (K=3)"] = top_k_unseen(centroid, seen_mask, C, item_ids, M)
            else:
                recs["Essence (K=3)"] = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
        else:
            for name in per_system_cutoff_recall:
                recs[name] = []

        for name, rec_list in recs.items():
            for k in CUTOFFS:
                per_system_cutoff_recall[name][k].append(recall_at_k(rec_list, test_items, k=k))
                lt = long_tail_recall_at_k(rec_list, test_items, lt_set, k=k)
                if lt is not None:
                    per_system_cutoff_lt[name][k].append(lt)

    rows = []
    for name in per_system_cutoff_recall:
        for k in CUTOFFS:
            r_mean = float(np.mean(per_system_cutoff_recall[name][k]))
            lt_vals = per_system_cutoff_lt[name][k]
            lt_mean = float(np.mean(lt_vals)) if lt_vals else float("nan")
            print(f"{name:<18} k={k:>2}  Recall={r_mean:.4f}  LT-Recall={lt_mean:.4f}")
            rows.append({"dataset": "amazon", "system": name, "k": k,
                        "recall@k": r_mean, "lt_recall@k": lt_mean,
                        "n_users": len(per_system_cutoff_recall[name][k]), "n_lt_users": len(lt_vals)})

    out_path = RESULTS_DIR / "multi_cutoff_amazon.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
