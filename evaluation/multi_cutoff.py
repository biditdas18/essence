"""
evaluation/multi_cutoff.py
------------------------------
Step 6: Recall@k / LT-Recall@k at k=5, 10, 20 for Essence and the three
recency baselines, Last.fm-1K. Retrieval is run ONCE at M=20 (the top-10
lists computed earlier were never cached to disk, only the resulting
metric values, so this can't reuse a cached ranking — but it avoids
rerunning retrieval three times by generating M=20 once and slicing).

Run:
    python evaluation/multi_cutoff.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import (
    essence_recommend, last_item_recommend, avg_last10_recommend, recency_weighted_recommend,
)
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

M = 20
CUTOFFS = [5, 10, 20]

SYSTEMS = {
    "Essence (K=3)": lambda uid, train_df, emb, M: essence_recommend(uid, train_df, emb, K=3, M=M),
    "Last-Item": last_item_recommend,
    "Avg-Last-10": avg_last10_recommend,
    "Recency-Weighted": recency_weighted_recommend,
}


def main():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as f:
        item_embedding_map = pickle.load(f)
    with open(DATA_DIR / "long_tail_ids.pkl", "rb") as f:
        long_tail_ids = pickle.load(f)

    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))

    rows = []
    for sys_name, fn in SYSTEMS.items():
        per_cutoff_recall = {k: [] for k in CUTOFFS}
        per_cutoff_lt = {k: [] for k in CUTOFFS}
        for uid in tqdm(all_users, desc=sys_name):
            recs20 = fn(uid, train_df, item_embedding_map, M)
            actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
            for k in CUTOFFS:
                per_cutoff_recall[k].append(recall_at_k(recs20, actual, k=k))
                lt = long_tail_recall_at_k(recs20, actual, long_tail_ids, k=k)
                if lt is not None:
                    per_cutoff_lt[k].append(lt)

        for k in CUTOFFS:
            r_mean = float(np.mean(per_cutoff_recall[k]))
            lt_mean = float(np.mean(per_cutoff_lt[k])) if per_cutoff_lt[k] else float("nan")
            rows.append({"dataset": "lastfm", "system": sys_name, "k": k,
                        "recall@k": r_mean, "lt_recall@k": lt_mean,
                        "n_users": len(per_cutoff_recall[k]), "n_lt_users": len(per_cutoff_lt[k])})
            print(f"{sys_name:<18} k={k:>2}  Recall={r_mean:.4f}  LT-Recall={lt_mean:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "multi_cutoff_lastfm.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
