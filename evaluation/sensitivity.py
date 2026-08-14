"""
evaluation/sensitivity.py
--------------------------
Two sensitivity analyses for Essence:

  1. K sweep: Essence at K = 2, 3, 4, 5 (fixed seed=42), report
     Recall@10 / LT-Recall@10 for each K.
  2. Seed variance: Essence at K=3 across 10 random K-means seeds,
     report mean and std of Recall@10 / LT-Recall@10 across seeds.

Saves per-run rows to results/sensitivity_results.csv.

Run:
    python evaluation/sensitivity.py
    python evaluation/sensitivity.py --users 99
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import essence_recommend
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

K_SWEEP = [2, 3, 4, 5]
SEED_SWEEP_K = 3
N_SEEDS = 10
SEEDS = list(range(10))  # 10 K-means random_state values for the seed-variance run
FIXED_SEED_FOR_K_SWEEP = 42
M = 10


def load_data():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as fh:
        item_embedding_map = pickle.load(fh)
    with open(DATA_DIR / "long_tail_ids.pkl", "rb") as fh:
        long_tail_ids = pickle.load(fh)
    return train_df, test_df, item_embedding_map, long_tail_ids


def run_one(train_df, test_df, item_embedding_map, long_tail_ids,
           users, K, seed):
    recalls, lt_recalls = [], []
    for user_id in users:
        actual = test_df[test_df["user_id"] == user_id]["track_id"].tolist()
        recs = essence_recommend(user_id, train_df, item_embedding_map, K=K, M=M, seed=seed)
        recalls.append(recall_at_k(recs, actual, k=M))
        lt = long_tail_recall_at_k(recs, actual, long_tail_ids, k=M)
        if lt is not None:
            lt_recalls.append(lt)
    return {
        "recall@10": float(np.mean(recalls)),
        "lt_recall@10": float(np.mean(lt_recalls)) if lt_recalls else float("nan"),
        "n_users": len(users),
        "n_lt_users": len(lt_recalls),
    }


def main():
    parser = argparse.ArgumentParser(description="Essence K-sensitivity and seed-variance analysis")
    parser.add_argument("--users", type=int, default=None,
                        help="Number of users to evaluate (default: all users present in both train/test)")
    args = parser.parse_args()

    train_df, test_df, item_embedding_map, long_tail_ids = load_data()
    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))
    users = all_users if args.users is None else all_users[:args.users]
    print(f"[sensitivity] Evaluating {len(users)} users\n")

    rows = []

    # --- K sweep -------------------------------------------------------------
    print(f"{'='*60}\n  K SWEEP (seed={FIXED_SEED_FOR_K_SWEEP})\n{'='*60}")
    print(f"  {'K':>3} {'Recall@10':>10} {'LT-Recall@10':>13}")
    for K in tqdm(K_SWEEP, desc="K sweep"):
        res = run_one(train_df, test_df, item_embedding_map, long_tail_ids,
                     users, K=K, seed=FIXED_SEED_FOR_K_SWEEP)
        print(f"  {K:>3} {res['recall@10']:>10.4f} {res['lt_recall@10']:>13.4f}")
        rows.append({
            "analysis": "K_sweep", "K": K, "seed": FIXED_SEED_FOR_K_SWEEP,
            "n_users": res["n_users"], "n_lt_users": res["n_lt_users"],
            "recall@10": res["recall@10"], "lt_recall@10": res["lt_recall@10"],
        })

    # --- Seed variance ---------------------------------------------------------
    print(f"\n{'='*60}\n  SEED VARIANCE (K={SEED_SWEEP_K}, {N_SEEDS} seeds)\n{'='*60}")
    print(f"  {'seed':>5} {'Recall@10':>10} {'LT-Recall@10':>13}")
    seed_recalls, seed_lt_recalls = [], []
    for seed in tqdm(SEEDS, desc="Seed variance"):
        res = run_one(train_df, test_df, item_embedding_map, long_tail_ids,
                     users, K=SEED_SWEEP_K, seed=seed)
        print(f"  {seed:>5} {res['recall@10']:>10.4f} {res['lt_recall@10']:>13.4f}")
        seed_recalls.append(res["recall@10"])
        seed_lt_recalls.append(res["lt_recall@10"])
        rows.append({
            "analysis": "seed_variance", "K": SEED_SWEEP_K, "seed": seed,
            "n_users": res["n_users"], "n_lt_users": res["n_lt_users"],
            "recall@10": res["recall@10"], "lt_recall@10": res["lt_recall@10"],
        })

    r_mean, r_std = float(np.mean(seed_recalls)), float(np.std(seed_recalls))
    lt_mean, lt_std = float(np.mean(seed_lt_recalls)), float(np.std(seed_lt_recalls))
    print(f"\n  Mean +/- std across {N_SEEDS} seeds:")
    print(f"    Recall@10:    {r_mean:.4f} +/- {r_std:.4f}")
    print(f"    LT-Recall@10: {lt_mean:.4f} +/- {lt_std:.4f}")
    rows.append({
        "analysis": "seed_variance_summary", "K": SEED_SWEEP_K, "seed": "mean",
        "n_users": len(users), "n_lt_users": "",
        "recall@10": r_mean, "lt_recall@10": lt_mean,
    })
    rows.append({
        "analysis": "seed_variance_summary", "K": SEED_SWEEP_K, "seed": "std",
        "n_users": len(users), "n_lt_users": "",
        "recall@10": r_std, "lt_recall@10": lt_std,
    })

    out_path = RESULTS_DIR / "sensitivity_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sensitivity] Saved to {out_path}\n")


if __name__ == "__main__":
    main()
