"""
experiments/amazon_books/sensitivity_amazon.py
------------------------------------------------
K-sensitivity and seed-variance analysis for Essence on Amazon Books,
mirroring evaluation/sensitivity.py's Last.fm analysis but reusing the
vectorized candidate-matrix approach from evaluate_amazon_peruser.py
(needed for speed at Amazon's scale: 61,727 candidate items).

  1. K sweep: Essence at K = 2, 3, 4, 5 (fixed seed=42).
  2. Seed variance: Essence at K=3 across 10 K-means seeds.

Uses Pass 1 (metadata) embeddings, matching evaluate_amazon_peruser.py.

Saves to:
  results/sensitivity_results_amazon.csv

Run:
    python experiments/amazon_books/sensitivity_amazon.py
"""

import pickle
import sys
import time
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
K_SWEEP = [2, 3, 4, 5]
FIXED_SEED_FOR_K_SWEEP = 42
SEED_SWEEP_K = 3
SEEDS = list(range(10))


def build_candidate_matrix(emb_meta: dict):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k=10):
    scores = C @ query_vec
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def run_essence_pass(train_df, test_map, lt_set, emb_meta, item_ids, C, item_index,
                     all_users, K, seed):
    recalls, lt_recalls = [], []
    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())

        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = [emb_meta[i] for i in train_items[-10:] if i in emb_meta]
            if recent_vecs:
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
            else:
                centroid = km.cluster_centers_[0].astype(np.float32)
            centroid /= (np.linalg.norm(centroid) + 1e-8)
            recs = top_k_unseen(centroid, seen_mask, C, item_ids, M)
        elif seen_vecs:
            user_vec = np.mean(seen_vecs, axis=0).astype(np.float32)
            user_vec /= (np.linalg.norm(user_vec) + 1e-8)
            recs = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
        else:
            recs = []

        recalls.append(recall_at_k(recs, test_items))
        ltr = lt_recall_at_k(recs, test_items, lt_set)
        if ltr is not None:
            lt_recalls.append(ltr)

    return {
        "recall@10": float(np.mean(recalls)),
        "lt_recall@10": float(np.mean(lt_recalls)) if lt_recalls else float("nan"),
        "n_users": len(all_users),
        "n_lt_users": len(lt_recalls),
    }


def main():
    print("[sensitivity_amazon] Loading data ...")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    from collections import defaultdict
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    print(f"[sensitivity_amazon] {len(all_users):,} users, {len(item_ids):,} items\n")

    rows = []

    print(f"{'='*60}\n  K SWEEP (seed={FIXED_SEED_FOR_K_SWEEP})\n{'='*60}")
    print(f"  {'K':>3} {'Recall@10':>10} {'LT-Recall@10':>13} {'time':>8}")
    for K in K_SWEEP:
        t0 = time.time()
        res = run_essence_pass(train_df, test_map, lt_set, emb_meta, item_ids, C,
                               item_index, all_users, K=K, seed=FIXED_SEED_FOR_K_SWEEP)
        dt = time.time() - t0
        print(f"  {K:>3} {res['recall@10']:>10.4f} {res['lt_recall@10']:>13.4f} {dt:>7.1f}s")
        rows.append({"analysis": "K_sweep", "K": K, "seed": FIXED_SEED_FOR_K_SWEEP,
                    "n_users": res["n_users"], "n_lt_users": res["n_lt_users"],
                    "recall@10": res["recall@10"], "lt_recall@10": res["lt_recall@10"]})

    print(f"\n{'='*60}\n  SEED VARIANCE (K={SEED_SWEEP_K}, {len(SEEDS)} seeds)\n{'='*60}")
    print(f"  {'seed':>5} {'Recall@10':>10} {'LT-Recall@10':>13} {'time':>8}")
    seed_recalls, seed_lt = [], []
    for seed in SEEDS:
        t0 = time.time()
        res = run_essence_pass(train_df, test_map, lt_set, emb_meta, item_ids, C,
                               item_index, all_users, K=SEED_SWEEP_K, seed=seed)
        dt = time.time() - t0
        print(f"  {seed:>5} {res['recall@10']:>10.4f} {res['lt_recall@10']:>13.4f} {dt:>7.1f}s")
        seed_recalls.append(res["recall@10"])
        seed_lt.append(res["lt_recall@10"])
        rows.append({"analysis": "seed_variance", "K": SEED_SWEEP_K, "seed": seed,
                    "n_users": res["n_users"], "n_lt_users": res["n_lt_users"],
                    "recall@10": res["recall@10"], "lt_recall@10": res["lt_recall@10"]})

    r_mean, r_std = float(np.mean(seed_recalls)), float(np.std(seed_recalls))
    lt_mean, lt_std = float(np.mean(seed_lt)), float(np.std(seed_lt))
    print(f"\n  Mean +/- std across {len(SEEDS)} seeds:")
    print(f"    Recall@10:    {r_mean:.4f} +/- {r_std:.4f}")
    print(f"    LT-Recall@10: {lt_mean:.4f} +/- {lt_std:.4f}")
    rows.append({"analysis": "seed_variance_summary", "K": SEED_SWEEP_K, "seed": "mean",
                "n_users": len(all_users), "n_lt_users": "",
                "recall@10": r_mean, "lt_recall@10": lt_mean})
    rows.append({"analysis": "seed_variance_summary", "K": SEED_SWEEP_K, "seed": "std",
                "n_users": len(all_users), "n_lt_users": "",
                "recall@10": r_std, "lt_recall@10": lt_std})

    import csv
    out_path = RESULTS_DIR / "sensitivity_results_amazon.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sensitivity_amazon] Saved to {out_path}\n")


if __name__ == "__main__":
    main()
