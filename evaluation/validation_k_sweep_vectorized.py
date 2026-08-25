"""
evaluation/validation_k_sweep_vectorized.py
----------------------------------------------
Tier-2 Step 6: validation-selected K for Amazon Books and MovieLens-25M,
using the same vectorized top-k-unseen pattern as
experiments/amazon_books/evaluate_amazon_peruser.py and
experiments/movielens/evaluate_movielens.py (per-item Python dict scoring,
as used by models.recommenders.essence_recommend, is too slow at these
catalog sizes -- 61,727 / 7,654 items).

Carves a validation split OUT OF TRAIN (not test): each user's existing
train.csv rows (already chronologically ordered per Phase 3 preprocessing)
are re-split 80/20 by row position, giving train' (first 80%) and a
validation set (last 20%) drawn only from what was previously train data.
Test is never touched. Essence is swept over K in {2,3,4,5,8} on
train'->validation.

Run:
    python evaluation/validation_k_sweep_vectorized.py --dataset amazon
    python evaluation/validation_k_sweep_vectorized.py --dataset movielens
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

K_SWEEP = [2, 3, 4, 5, 8]
SEED = 42
M = 10
VAL_FRACTION = 0.20

DATASETS = {
    "amazon": {
        "proc_dir": BASE_DIR / "data" / "amazon_processed",
        "train_file": "train.csv",
        "emb_file": "embeddings_metadata.pkl",
        "has_timestamp": False,
    },
    "movielens": {
        "proc_dir": BASE_DIR / "data" / "movielens_processed",
        "train_file": "train.csv",
        "emb_file": "embeddings_metadata.pkl",
        "has_timestamp": False,
    },
}


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


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def row_order_split(df: pd.DataFrame, frac: float):
    """Split each user's rows (already chronological by construction) 80/20 by position."""
    tr_rows, val_rows = [], []
    for uid, group in df.groupby("user_id", sort=False):
        n = len(group)
        n_val = max(1, round(n * frac))
        n_tr = n - n_val
        if n_tr < 1:
            continue
        tr_rows.append(group.iloc[:n_tr])
        val_rows.append(group.iloc[n_tr:])
    return pd.concat(tr_rows).reset_index(drop=True), pd.concat(val_rows).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--k-list", default=None, help="comma-separated K values, overrides K_SWEEP")
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    k_values = [int(k) for k in args.k_list.split(",")] if args.k_list else K_SWEEP

    t0 = time.time()
    train_df = pd.read_csv(cfg["proc_dir"] / cfg["train_file"])
    with open(cfg["proc_dir"] / cfg["emb_file"], "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    print(f"[val_k_sweep:{args.dataset}] {len(item_ids):,} catalog items")

    train_p, val_df = row_order_split(train_df, VAL_FRACTION)
    val_users = sorted(set(train_p["user_id"].unique()) & set(val_df["user_id"].unique()))
    print(f"[val_k_sweep:{args.dataset}] {len(val_users):,} users with both train' and validation data")

    val_map = {}
    for uid, group in val_df.groupby("user_id"):
        val_map[uid] = set(group["item_id"])

    results = []
    for K in k_values:
        recalls = []
        cluster_sizes = []  # items-per-cluster, users that actually clustered only
        n_clustered = 0
        for uid in tqdm(val_users, desc=f"K={K}"):
            user_rows = train_p[train_p["user_id"] == uid]
            train_items = list(user_rows["item_id"])
            seen_set = set(train_items)
            test_items = val_map.get(uid, set())

            seen_mask = np.zeros(len(item_ids), dtype=bool)
            for iid in seen_set:
                idx = item_index.get(iid)
                if idx is not None:
                    seen_mask[idx] = True

            seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
            if len(seen_vecs) < K:
                if seen_vecs:
                    query = normalized(np.mean(seen_vecs, axis=0))
                    recs = top_k_unseen(query, seen_mask, C, item_ids, M)
                else:
                    recs = []
            else:
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
                km.fit(np.array(seen_vecs))
                cluster_sizes.append(len(seen_vecs) / K)
                n_clustered += 1
                recent_vecs = seen_vecs[-10:]
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                recs = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)

            recalls.append(recall_at_k(recs, test_items))

        mean_recall = float(np.mean(recalls))
        mean_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else float("nan")
        frac_clustered = n_clustered / len(val_users)
        elapsed_k = time.time() - t0
        print(f"  K={K}: val Recall@10 = {mean_recall:.4f} | mean items/cluster = {mean_cluster_size:.2f} "
              f"({frac_clustered*100:.1f}% users clustered) (elapsed so far: {elapsed_k:.1f}s)")
        results.append({
            "dataset": args.dataset, "K": K, "val_recall@10": mean_recall,
            "mean_items_per_cluster": mean_cluster_size, "frac_users_clustered": frac_clustered,
            "n_users": len(val_users),
        })

    out_path = BASE_DIR / "results" / f"validation_k_sweep_{args.dataset}.csv"
    new_df = pd.DataFrame(results)
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        for col in ["mean_items_per_cluster", "frac_users_clustered"]:
            if col not in old_df.columns:
                old_df[col] = float("nan")
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["dataset", "K"], keep="last").sort_values("K")
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    out = combined
    best = out.loc[out["val_recall@10"].idxmax()]
    print(f"\n[val_k_sweep:{args.dataset}] Selected K = {int(best['K'])} (val Recall@10={best['val_recall@10']:.4f})")
    print(f"[val_k_sweep:{args.dataset}] Done in {time.time()-t0:.1f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
