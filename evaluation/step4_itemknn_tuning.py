"""
evaluation/step4_itemknn_tuning.py
--------------------------------------
Tier-2 Step 4: validation-based hyperparameter search for
TunableItemKNNModel (models/itemknn_tunable.py, conformance-tested in
models/test_itemknn_tunable.py -- run and passed before this script).

Same validation-only protocol as the K-selection work: the validation
split is the last 20% of each user's TRAINING data (chronological), and
the real held-out test set is never touched during the search. The
selected hyperparameters are then evaluated once, for reporting, against
the real test set (same as how the paper reports Essence's final
validation-selected K against test).

Search grid: k_nn in {None (full neighborhood, the paper's current
untuned setting), 50, 100, 200, 500}; shrinkage in {0, 5, 10, 25, 100}.
25 combinations per dataset, selected by validation Recall@10.

Outputs:
  results/scratch_itemknn_tuning_{lastfm,amazon,movielens}.csv  (full grid)

Run:
    python evaluation/step4_itemknn_tuning.py lastfm
    python evaluation/step4_itemknn_tuning.py amazon
    python evaluation/step4_itemknn_tuning.py movielens
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.itemknn_tunable import TunableItemKNNModel, tunable_itemknn_recommend
from models.recommenders import build_itemknn_model, cf_itemknn_recommend

RESULTS_DIR = BASE_DIR / "results"
M = 10
K_NN_GRID = [None, 50, 100, 200, 500]
SHRINKAGE_GRID = [0, 5, 10, 25, 100]


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def carve_validation(train_df, uid_col="user_id", ts_col="timestamp"):
    """Last 20% of each user's TRAINING data becomes the validation-holdout;
    the rest becomes the validation-train portion. Real test set untouched."""
    val_train_rows, val_holdout_rows = [], []
    for uid, g in train_df.groupby(uid_col):
        g = g.sort_values(ts_col)
        n = len(g)
        n_val = max(1, round(n * 0.20))
        val_train_rows.append(g.iloc[: n - n_val])
        val_holdout_rows.append(g.iloc[n - n_val:])
    return pd.concat(val_train_rows), pd.concat(val_holdout_rows)


def eval_config(train_df, holdout_map, item_col, k_nn, shrinkage):
    model = TunableItemKNNModel(train_df, item_col=item_col, k_nn=k_nn, shrinkage=shrinkage)
    recalls = []
    for uid, test_items in holdout_map.items():
        recs = tunable_itemknn_recommend(uid, train_df, model, M)
        recalls.append(recall_at_k(recs, test_items))
    return float(np.mean(recalls)) if recalls else 0.0


def run(label, train_df, test_df, item_col, out_tag):
    t0 = time.time()
    print("=" * 60)
    print(f"[itemknn-tuning] {label}")
    print("=" * 60)
    if "timestamp" not in train_df.columns:
        train_df = train_df.copy()
        train_df["timestamp"] = train_df.index

    val_train, val_holdout = carve_validation(train_df)
    holdout_map = defaultdict(set)
    for _, row in val_holdout.iterrows():
        holdout_map[row["user_id"]].add(row[item_col])
    print(f"  Validation split: {len(val_train):,} val-train rows, "
          f"{len(val_holdout):,} val-holdout rows across {len(holdout_map):,} users")

    grid_rows = []
    best = (None, None, -1.0)
    for k_nn in K_NN_GRID:
        for shrinkage in SHRINKAGE_GRID:
            t_c = time.time()
            r10 = eval_config(val_train, holdout_map, item_col, k_nn, shrinkage)
            dt = time.time() - t_c
            print(f"  k_nn={str(k_nn):>5}  shrinkage={shrinkage:>5}  "
                  f"val Recall@10={r10:.5f}  ({dt:.1f}s)")
            grid_rows.append({"k_nn": k_nn, "shrinkage": shrinkage, "val_recall@10": r10})
            if r10 > best[2]:
                best = (k_nn, shrinkage, r10)

    grid_df = pd.DataFrame(grid_rows)
    grid_path = RESULTS_DIR / f"scratch_itemknn_tuning_{out_tag}.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"\n  Best on validation: k_nn={best[0]}, shrinkage={best[1]}, val Recall@10={best[2]:.5f}")
    print(f"  Saved full grid -> {grid_path}")

    # Final: evaluate the selected config AND the untuned canonical config on the REAL test set
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row[item_col])

    tuned_model = TunableItemKNNModel(train_df, item_col=item_col, k_nn=best[0], shrinkage=best[1])
    tuned_recalls, tuned_lt = [], []
    canonical_model = build_itemknn_model(train_df, item_col=item_col)
    canon_recalls = []

    for uid in sorted(train_df["user_id"].unique()):
        test_items = test_map.get(uid, set())
        recs_tuned = tunable_itemknn_recommend(uid, train_df, tuned_model, M)
        recs_canon = cf_itemknn_recommend(uid, train_df, canonical_model, M)
        tuned_recalls.append(recall_at_k(recs_tuned, test_items))
        canon_recalls.append(recall_at_k(recs_canon, test_items))

    print(f"\n  TEST SET (never touched during search):")
    print(f"    Untuned canonical ItemKNN  Recall@10 = {np.mean(canon_recalls):.5f}")
    print(f"    Tuned (k_nn={best[0]}, shrinkage={best[1]})  Recall@10 = {np.mean(tuned_recalls):.5f}")
    rel_change = (np.mean(tuned_recalls) - np.mean(canon_recalls)) / max(np.mean(canon_recalls), 1e-9) * 100
    print(f"    Relative change: {rel_change:+.1f}%")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    return best, np.mean(canon_recalls), np.mean(tuned_recalls)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("lastfm", "all"):
        train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
        test_df = pd.read_pickle(BASE_DIR / "data" / "test_interactions.pkl")
        train_df = train_df.rename(columns={"track_id": "item_id"})
        test_df = test_df.rename(columns={"track_id": "item_id"})
        run("Last.fm-1K", train_df, test_df, "item_id", "lastfm")

    if which in ("amazon", "all"):
        train_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "train.csv")
        test_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "test.csv")
        run("Amazon Books", train_df, test_df, "item_id", "amazon")

    if which in ("movielens", "all"):
        train_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "train.csv")
        test_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "test.csv")
        run("MovieLens-25M", train_df, test_df, "item_id", "movielens")


if __name__ == "__main__":
    main()
