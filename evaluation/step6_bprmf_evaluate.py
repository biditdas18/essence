"""
evaluation/step6_bprmf_evaluate.py
-------------------------------------
Tier-2 Step 6: evaluate BPR-MF (models/bpr_mf.py) as a new collaborative
baseline against Essence, on all three datasets' EXISTING canonical
train/test splits (no new data built -- this step adds one new system to
the existing evaluation, it does not change any split).

PREREQUISITE, already run and passed: models/test_bpr_mf.py. Do not run
this script's real-dataset comparisons without that having passed first
-- this script imports and calls the same BPRMF class, so a failing
conformance check means these numbers cannot be trusted.

Hyperparameters (fixed, not tuned -- consistent with how the paper treats
ItemKNN as an untuned baseline; Step 4 handles hyperparameter tuning
separately and only for ItemKNN):
  n_factors=32, lr=0.05, reg=0.01, n_epochs=30, seed=42

Outputs:
  results/scratch_bprmf_{lastfm,amazon,movielens}.csv   (per-user, BPR-MF only)
  results/scratch_bprmf_bootstrap_{lastfm,amazon,movielens}.csv

Run:
    python evaluation/step6_bprmf_evaluate.py lastfm
    python evaluation/step6_bprmf_evaluate.py amazon
    python evaluation/step6_bprmf_evaluate.py movielens
"""
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.bpr_mf import BPRMF, bpr_recommend

RESULTS_DIR = BASE_DIR / "results"
M = 10
N_FACTORS = 32
LR = 0.05
REG = 0.01
N_EPOCHS = 30
SEED = 42


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def run(label, train_df, test_df, lt_set, item_col, essence_merged_path, out_tag):
    t0 = time.time()
    print("=" * 60)
    print(f"[bprmf] {label}")
    print("=" * 60)

    all_users = sorted(train_df["user_id"].unique())
    all_items = sorted(set(train_df[item_col]) | set(test_df[item_col]))
    user_idx = {u: i for i, u in enumerate(all_users)}
    item_idx = {t: i for i, t in enumerate(all_items)}
    print(f"  Users: {len(all_users):,}  Items: {len(all_items):,}")

    user_pos_items = defaultdict(list)
    for _, row in train_df.iterrows():
        ui, ii = user_idx.get(row["user_id"]), item_idx.get(row[item_col])
        if ui is not None and ii is not None:
            user_pos_items[ui].append(ii)

    n_train_interactions = sum(len(v) for v in user_pos_items.values())
    print(f"  Training BPR-MF: n_factors={N_FACTORS}, epochs={N_EPOCHS}, "
          f"{n_train_interactions:,} positive interactions ...")
    model = BPRMF(len(all_users), len(all_items), n_factors=N_FACTORS, lr=LR, reg=REG, seed=SEED)
    t_train = time.time()
    model.fit(dict(user_pos_items), n_epochs=N_EPOCHS, seed=SEED, verbose=True)
    print(f"  Training done in {time.time()-t_train:.1f}s")

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row[item_col])

    rows = []
    for uid in all_users:
        u = user_idx[uid]
        seen_idx = set(item_idx[t] for t in train_df[train_df["user_id"] == uid][item_col] if t in item_idx)
        test_items = test_map.get(uid, set())
        recs = bpr_recommend(u, seen_idx, model, all_items, M)
        rows.append({
            "user_id": uid, "system": "BPR-MF",
            "recall@10": recall_at_k(recs, test_items),
            "long_tail_recall@10": lt_recall_at_k(recs, test_items, lt_set),
        })
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"scratch_bprmf_{out_tag}.csv"
    df.to_csv(out_path, index=False)
    r10, ltr = df["recall@10"].mean(), df["long_tail_recall@10"].dropna().mean()
    print(f"  BPR-MF: Recall@10={r10:.4f}  LT-Recall@10={ltr:.4f}")
    print(f"  Saved -> {out_path}")

    # Merge with existing Essence + baselines for a direct bootstrap comparison
    essence_all = pd.read_csv(essence_merged_path)
    merged = pd.concat([essence_all, df], ignore_index=True)
    merged_path = RESULTS_DIR / f"scratch_bprmf_merged_{out_tag}.csv"
    merged.to_csv(merged_path, index=False)

    bootstrap_path = RESULTS_DIR / f"scratch_bprmf_bootstrap_{out_tag}.csv"
    subprocess.run(
        [sys.executable, "evaluation/paired_bootstrap.py",
         "--input", str(merged_path), "--label", f"{label} + BPR-MF",
         "--output", str(bootstrap_path)],
        cwd=str(BASE_DIR), check=True,
    )
    print(f"  Saved bootstrap -> {bootstrap_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    return out_path


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("lastfm", "all"):
        train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
        test_df = pd.read_pickle(BASE_DIR / "data" / "test_interactions.pkl")
        import pickle
        with open(BASE_DIR / "data" / "long_tail_ids.pkl", "rb") as f:
            lt_set = pickle.load(f)
        run("Last.fm-1K", train_df, test_df, lt_set, "track_id",
            RESULTS_DIR / "scratch_evaluation_results_K10_lastfm_merged.csv", "lastfm")

    if which in ("amazon", "all"):
        train_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "train.csv")
        test_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "test.csv")
        lt_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "longtail_items.csv")
        run("Amazon Books", train_df, test_df, set(lt_df["item_id"]), "item_id",
            RESULTS_DIR / "scratch_evaluation_results_K10_amazon_merged.csv", "amazon")

    if which in ("movielens", "all"):
        train_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "train.csv")
        test_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "test.csv")
        lt_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "longtail_items.csv")
        run("MovieLens-25M", train_df, test_df, set(lt_df["item_id"]), "item_id",
            RESULTS_DIR / "scratch_evaluation_results_K15_movielens_merged.csv", "movielens")


if __name__ == "__main__":
    main()
