"""
experiments/amazon_books/shortcut_check_amazon.py
-----------------------------------------------------
Author-shortcut check for Essence on Amazon Books — the book-domain
analogue of evaluation/shortcut_check.py's artist check (Amazon has no
artist field; "author" is the equivalent identity signal for books).

  1. Shortcut fraction: for each user's Essence top-10 recs, what
     fraction share an author with an item in the user's active cluster?
  2. Author-holdout variant: rerun Essence excluding candidates that
     share an author with the active cluster; report Recall@10 /
     LT-Recall@10.

Uses Pass 1 (metadata) embeddings, matching evaluate_amazon_peruser.py.

Saves per-user rows to:
  results/shortcut_analysis_amazon.csv
  results/shortcut_analysis_amazon_summary.csv

Run:
    python experiments/amazon_books/shortcut_check_amazon.py
"""

import csv
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
K = 3
SEED = 42


def build_candidate_matrix(emb_meta: dict):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def main():
    print("[shortcut_check_amazon] Loading data ...")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    item_meta = pd.read_csv(PROC_DIR / "item_meta.csv")
    item_author_map = dict(zip(item_meta["item_id"], item_meta["author"]))

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    print(f"[shortcut_check_amazon] {len(all_users):,} users, {len(item_ids):,} items (K={K}, M={M})\n")

    rows = []
    total_recs = 0
    total_shared = 0

    for uid in tqdm(all_users, desc="Users"):
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())

        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        seen_items_valid = [i for i in train_items if i in emb_meta]
        seen_vecs = [emb_meta[i] for i in seen_items_valid]

        if len(seen_vecs) < K:
            # Fallback: no clustering possible; active "cluster" = full history.
            active_authors = {item_author_map.get(i) for i in seen_items_valid}
            if seen_vecs:
                user_vec = np.mean(seen_vecs, axis=0).astype(np.float32)
                user_vec /= (np.linalg.norm(user_vec) + 1e-8)
                scores = C @ user_vec
                scores[seen_mask] = -2.0
                top_idx = np.lexsort((np.arange(len(scores)), -scores))[:M]
                recs = [item_ids[i] for i in top_idx]

                holdout_mask = seen_mask.copy()
                for iid, idx in item_index.items():
                    if item_author_map.get(iid) in active_authors:
                        holdout_mask[idx] = True
                hscores = C @ user_vec
                hscores[holdout_mask] = -2.0
                htop_idx = np.lexsort((np.arange(len(hscores)), -hscores))[:M]
                holdout_recs = [item_ids[i] for i in htop_idx]
            else:
                recs, holdout_recs = [], []
        else:
            km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = [emb_meta[i] for i in train_items[-10:] if i in emb_meta]
            if recent_vecs:
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                active_idx = int(np.argmin(dists))
            else:
                active_idx = 0
            centroid = km.cluster_centers_[active_idx].astype(np.float32)
            centroid /= (np.linalg.norm(centroid) + 1e-8)

            labels = km.labels_
            active_cluster_items = [seen_items_valid[i] for i in range(len(seen_items_valid))
                                    if labels[i] == active_idx]
            active_authors = {item_author_map.get(i) for i in active_cluster_items}

            scores = C @ centroid
            scores[seen_mask] = -2.0
            top_idx = np.lexsort((np.arange(len(scores)), -scores))[:M]
            recs = [item_ids[i] for i in top_idx]

            holdout_mask = seen_mask.copy()
            for iid, idx in item_index.items():
                if item_author_map.get(iid) in active_authors:
                    holdout_mask[idx] = True
            hscores = C @ centroid
            hscores[holdout_mask] = -2.0
            htop_idx = np.lexsort((np.arange(len(hscores)), -hscores))[:M]
            holdout_recs = [item_ids[i] for i in htop_idx]

        n_recs = len(recs)
        n_shared = sum(1 for iid in recs if item_author_map.get(iid) in active_authors) if recs else 0
        total_recs += n_recs
        total_shared += n_shared

        holdout_r10 = recall_at_k(holdout_recs, test_items)
        holdout_lt = lt_recall_at_k(holdout_recs, test_items, lt_set)

        rows.append({
            "user_id": uid,
            "n_recs": n_recs,
            "n_shared_author": n_shared,
            "shortcut_fraction": n_shared / n_recs if n_recs else "",
            "holdout_recall@10": holdout_r10,
            "holdout_long_tail_recall@10": "" if holdout_lt is None else holdout_lt,
        })

    pooled_fraction = total_shared / total_recs if total_recs else float("nan")
    fracs = [r["shortcut_fraction"] for r in rows if r["shortcut_fraction"] != ""]
    macro_fraction = float(np.mean(fracs)) if fracs else float("nan")
    holdout_recalls = [r["holdout_recall@10"] for r in rows]
    holdout_lts = [r["holdout_long_tail_recall@10"] for r in rows if r["holdout_long_tail_recall@10"] != ""]

    print(f"\n{'='*60}")
    print("  AUTHOR SHORTCUT CHECK — Essence (K=3), Amazon Books")
    print(f"{'='*60}")
    print(f"  Recs sharing an author with the active cluster:")
    print(f"    pooled fraction: {pooled_fraction:.4f}  ({total_shared}/{total_recs})")
    print(f"    macro fraction:  {macro_fraction:.4f}")
    print(f"\n  Author-holdout variant:")
    print(f"    Recall@10:    {np.mean(holdout_recalls):.4f}")
    print(f"    LT-Recall@10: {np.mean(holdout_lts):.4f}  (n={len(holdout_lts)})")
    print()

    out_path = RESULTS_DIR / "shortcut_analysis_amazon.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[shortcut_check_amazon] Saved per-user rows to {out_path}")

    summary_path = RESULTS_DIR / "shortcut_analysis_amazon_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["pooled_shortcut_fraction", f"{pooled_fraction:.6f}"])
        w.writerow(["macro_shortcut_fraction", f"{macro_fraction:.6f}"])
        w.writerow(["holdout_recall@10", f"{np.mean(holdout_recalls):.6f}"])
        w.writerow(["holdout_lt_recall@10", f"{np.mean(holdout_lts):.6f}"])
        w.writerow(["n_users", len(all_users)])
    print(f"[shortcut_check_amazon] Saved summary to {summary_path}\n")


if __name__ == "__main__":
    main()
