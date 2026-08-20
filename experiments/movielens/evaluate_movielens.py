"""
experiments/movielens/evaluate_movielens.py
-----------------------------------------------
Step 11c: full baseline suite on MovieLens-25M (2,000-user subsample),
modeled directly on evaluate_amazon_peruser.py's vectorized pattern
(same 8 systems: Random, Popularity, CF-ItemKNN, Content, Last-Item,
Avg-Last-10, Recency-Weighted, Essence). MIND/ComiRec run separately via
experiments/mind_comirec/train.py --dataset movielens (added to dataset.py).

Saves:
  experiments/movielens/results_movielens_peruser.csv
  experiments/movielens/results_movielens_aggregate.csv

Run:
    python experiments/movielens/evaluate_movielens.py
"""

import csv
import hashlib
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
OUT_DIR = Path(__file__).parent

M = 10
K = 3
RECENCY_N = 10
RECENCY_DECAY = 0.9


def _stable_user_seed(user_id) -> int:
    return int.from_bytes(hashlib.md5(str(user_id).encode()).digest()[:4], "big")


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k=10):
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


def main():
    print("=" * 60)
    print("MovieLens-25M (2,000-user subsample) -- 8-system evaluation")
    print("=" * 60)
    t_total = time.time()

    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])

    all_users = sorted(train_df["user_id"].unique())
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    print(f"Users: {len(all_users):,}  LT items: {len(lt_set):,}")

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)
    print(f"Candidate items (with embeddings): {n_items:,}")

    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)
    itemknn = build_itemknn_model(train_df, item_col="item_id")

    systems = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)",
              "Last-Item", "Avg-Last-10", "Recency-Weighted", "Essence (K=3)"]
    recall = defaultdict(list)
    lt_recall = defaultdict(list)
    per_user_rows = []

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

        unseen_pool = [i for i in item_ids if i not in seen_set]
        rng_u = np.random.default_rng(_stable_user_seed(uid))
        recs_random = rng_u.choice(unseen_pool, size=min(M, len(unseen_pool)), replace=False).tolist()

        recs_pop = [iid for iid in popularity.index if iid not in seen_set][:M]
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            user_vec = normalized(np.mean(seen_vecs, axis=0))
            recs_content = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
            recs_last_item = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs_avg_last10 = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs_recency_weighted = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)

            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=42, n_init=10)
                km.fit(np.array(seen_vecs))
                if recent_vecs:
                    recent_mean = np.mean(recent_vecs, axis=0)
                    dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                    centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                else:
                    centroid = km.cluster_centers_[0].astype(np.float32)
                recs_essence = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs_essence = recs_content
        else:
            recs_content = recs_last_item = recs_avg_last10 = recs_recency_weighted = recs_essence = recs_pop

        for name, recs in zip(systems, [recs_random, recs_pop, recs_knn, recs_content,
                                        recs_last_item, recs_avg_last10, recs_recency_weighted, recs_essence]):
            r10v = recall_at_k(recs, test_items, k=M)
            recall[name].append(r10v)
            ltr = long_tail_recall_at_k(recs, test_items, lt_set, k=M)
            if ltr is not None:
                lt_recall[name].append(ltr)
            per_user_rows.append({"user_id": uid, "system": name, "recall@10": r10v,
                                  "long_tail_recall@10": "" if ltr is None else ltr})

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s")

    rows_out = []
    print(f"\n{'System':<20} {'R@10':>8} {'LT-R@10':>10}")
    for name in systems:
        r10 = float(np.mean(recall[name]))
        ltr = float(np.mean(lt_recall[name])) if lt_recall[name] else 0.0
        print(f"{name:<20} {r10:>8.4f} {ltr:>10.4f}")
        rows_out.append({"dataset": "movielens", "system": name, "Recall@10": f"{r10:.4f}", "LT-Recall@10": f"{ltr:.4f}"})

    with open(OUT_DIR / "results_movielens_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)

    with open(OUT_DIR / "results_movielens_peruser.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
        w.writeheader()
        w.writerows(per_user_rows)
    print(f"\nSaved to {OUT_DIR / 'results_movielens_peruser.csv'} and results_movielens_aggregate.csv")


if __name__ == "__main__":
    main()
