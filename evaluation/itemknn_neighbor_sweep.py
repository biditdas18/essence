"""
evaluation/itemknn_neighbor_sweep.py
----------------------------------------
Step 5 (fairness parity with Essence's K-sweep): sweep ItemKNN's
neighbor count N on Last.fm and report whether the current default was
near-optimal.

IMPORTANT PREMISE CHECK: the canonical ItemKNNModel in
models/recommenders.py (cf_itemknn_recommend) does NOT truncate to a
top-N neighbor set at all -- it aggregates cosine similarity over ALL of
a user's seen items (score(i) = sum_{j in seen(u)} sim(i,j), full
aggregate scoring per Sarwar et al. 2001's simplest variant). There is
no "current default N" to sweep; effectively N = unbounded. This script
adds an explicit top-N neighbor truncation as a standalone variant
(mirroring the classic textbook ItemKNN restriction: each item's score
only aggregates over its N most-similar OTHER items, not all of them)
purely for this sensitivity analysis, WITHOUT modifying
models/recommenders.py or any canonical baseline used elsewhere in this
repo. "unrestricted" (N >= n_items) reproduces the canonical algorithm
exactly and is included as a sanity-check endpoint.

Run:
    python evaluation/itemknn_neighbor_sweep.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import build_itemknn_model
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_VALUES = [5, 10, 20, 50, 100, None]  # None = unrestricted (canonical behavior)


def truncate_topn(S: csr_matrix, N: int) -> csr_matrix:
    """Row-wise top-N truncation of a sparse similarity matrix, self-similarity excluded."""
    S = S.tolil(copy=True)
    n = S.shape[0]
    for i in range(n):
        row = S.rows[i]
        data = S.data[i]
        if not row:
            continue
        # exclude self-similarity
        pairs = [(idx, val) for idx, val in zip(row, data) if idx != i]
        if len(pairs) > N:
            pairs.sort(key=lambda p: -p[1])
            pairs = pairs[:N]
        S.rows[i] = [p[0] for p in pairs]
        S.data[i] = [p[1] for p in pairs]
    return S.tocsr()


def main():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(DATA_DIR / "long_tail_ids.pkl", "rb") as f:
        long_tail_ids = pickle.load(f)

    model = build_itemknn_model(train_df, item_col="track_id")
    S_full = model.R_norm @ model.R_norm.T  # sparse item-item similarity, includes self-sim
    n_items = S_full.shape[0]
    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))

    print(f"[itemknn_sweep] n_items={n_items} n_users={len(all_users)}")
    print(f"{'N':>12} {'Recall@10':>10} {'LT-Recall@10':>13}")

    rows = []
    for N in N_VALUES:
        S_n = S_full if N is None else truncate_topn(S_full, N)

        recalls, lt_recalls = [], []
        for uid in all_users:
            seen = set(train_df[train_df["user_id"] == uid]["track_id"])
            seen_indices = [model.item_idx[t] for t in seen if t in model.item_idx]
            if not seen_indices:
                continue
            seen_indicator = np.zeros(n_items)
            seen_indicator[seen_indices] = 1.0
            scores = np.asarray(S_n @ seen_indicator).ravel()
            for idx in seen_indices:
                scores[idx] = -np.inf
            top_indices = np.lexsort((np.arange(n_items), -scores))[:M]
            recs = [model.all_items[i] for i in top_indices if scores[i] > -np.inf][:M]

            actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
            recalls.append(recall_at_k(recs, actual, k=M))
            lt = long_tail_recall_at_k(recs, actual, long_tail_ids, k=M)
            if lt is not None:
                lt_recalls.append(lt)

        r_mean = float(np.mean(recalls))
        lt_mean = float(np.mean(lt_recalls)) if lt_recalls else float("nan")
        label = "unrestricted" if N is None else str(N)
        print(f"{label:>12} {r_mean:>10.4f} {lt_mean:>13.4f}")
        rows.append({"N": label, "recall@10": r_mean, "lt_recall@10": lt_mean,
                    "n_users": len(recalls), "n_lt_users": len(lt_recalls)})

    out_path = RESULTS_DIR / "itemknn_neighbor_sweep.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
