"""
evaluation/validation_k_sweep_lastfm.py
-----------------------------------------
Tier-2 Step 6: validation-selected K for Last.fm-1K.

Carves a validation split OUT OF TRAIN (not test): each user's existing
train_interactions.pkl history is chronologically re-split 80/20, giving
train' (first 80%) and a validation set (last 20%) drawn only from what
was previously train data. Test is never touched here. Essence is swept
over K in {2,3,4,5,8} on train'->validation, exactly mirroring the
existing chrono_split pattern in experiments/phase1_chrono_split.py.

Run:
    python evaluation/validation_k_sweep_lastfm.py
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import essence_recommend
from evaluation.evaluate import recall_at_k

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

K_SWEEP = [2, 3, 4, 5, 8]
SEED = 42
M = 10
VAL_FRACTION = 0.20


def chrono_split(df: pd.DataFrame, frac: float):
    tr_rows, val_rows = [], []
    for uid, group in df.groupby("user_id", sort=False):
        g = group.sort_values("timestamp")
        n = len(g)
        n_val = max(1, round(n * frac))
        n_tr = n - n_val
        if n_tr < 1:
            continue
        tr_rows.append(g.iloc[:n_tr])
        val_rows.append(g.iloc[n_tr:])
    return pd.concat(tr_rows).reset_index(drop=True), pd.concat(val_rows).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-list", default=None, help="comma-separated K values, overrides K_SWEEP")
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_list.split(",")] if args.k_list else K_SWEEP

    t0 = time.time()
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as fh:
        item_embedding_map = pickle.load(fh)

    train_p, val_df = chrono_split(train_df, VAL_FRACTION)
    val_users = sorted(set(train_p["user_id"].unique()) & set(val_df["user_id"].unique()))
    print(f"[val_k_sweep_lastfm] {len(val_users)} users with both train' and validation data")

    results = []
    for K in k_values:
        recalls = []
        cluster_sizes = []
        n_clustered = 0
        for uid in tqdm(val_users, desc=f"K={K}"):
            actual = val_df[val_df["user_id"] == uid]["track_id"].tolist()
            recs = essence_recommend(uid, train_p, item_embedding_map, K=K, M=M, seed=SEED)
            recalls.append(recall_at_k(recs, actual, k=M))
            n_items = len(train_p[train_p["user_id"] == uid])
            if n_items >= K:
                cluster_sizes.append(n_items / K)
                n_clustered += 1
        mean_recall = float(np.mean(recalls))
        mean_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else float("nan")
        frac_clustered = n_clustered / len(val_users)
        print(f"  K={K}: val Recall@10 = {mean_recall:.4f} | mean items/cluster = {mean_cluster_size:.2f} "
              f"({frac_clustered*100:.1f}% users clustered)")
        results.append({
            "dataset": "lastfm", "K": K, "val_recall@10": mean_recall,
            "mean_items_per_cluster": mean_cluster_size, "frac_users_clustered": frac_clustered,
            "n_users": len(val_users),
        })

    out_path = BASE_DIR / "results" / "validation_k_sweep_lastfm.csv"
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
    print(f"\n[val_k_sweep_lastfm] Selected K = {int(best['K'])} (val Recall@10={best['val_recall@10']:.4f})")
    print(f"[val_k_sweep_lastfm] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
