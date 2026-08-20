"""
experiments/movielens/shortcut_check_movielens.py
------------------------------------------------------
Step 11c.g: shortcut check (Step E pattern) for MovieLens. Movies don't
have an "artist" field the way tracks/books do; the closest equivalent
signal available in this dataset is GENRE (from movies.csv) -- stated
explicitly here per the task's instruction to use "whatever is the
closest equivalent." A movie's genre set is used as its identity tag,
and "sharing an artist" becomes "sharing at least one genre" with the
active cluster.

Checks whether Essence's top-10 recs are dominated by same-genre
candidates relative to the user's active cluster, and reports the
genre-holdout variant (excluding same-genre candidates) Recall@10 /
LT-Recall@10.

Saves:
  results/shortcut_analysis_movielens.csv
  results/shortcut_analysis_movielens_summary.csv

Run:
    python experiments/movielens/shortcut_check_movielens.py
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
RAW_DIR = BASE_DIR / "data" / "movielens_raw" / "ml-25m"
RESULTS_DIR = BASE_DIR / "results"

M = 10
K = 3
SEED = 42


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def lt_recall_at_k(recs, actual, lt_set, k=10):
    actual_lt = [i for i in actual if i in lt_set]
    if not actual_lt:
        return None
    return len(set(recs[:k]) & set(actual_lt)) / len(actual_lt)


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])

    movies = pd.read_csv(RAW_DIR / "movies.csv")
    item_genres = {}
    for _, row in movies.iterrows():
        g = row["genres"]
        item_genres[row["movieId"]] = set(g.split("|")) if pd.notna(g) and g != "(no genres listed)" else set()

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    rows = []
    total_recs, total_shared = 0, 0

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

        seen_valid = [i for i in train_items if i in emb_meta]
        vecs = [emb_meta[i] for i in seen_valid]
        if len(vecs) < K:
            continue

        km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
        km.fit(np.array(vecs))
        recent = train_items[-10:]
        recent_vecs = [emb_meta[i] for i in recent if i in emb_meta]
        if recent_vecs:
            recent_mean = np.mean(recent_vecs, axis=0)
            dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
            active_idx = int(np.argmin(dists))
        else:
            active_idx = 0
        centroid = km.cluster_centers_[active_idx].astype(np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-8)

        labels = km.labels_
        active_cluster_items = [seen_valid[i] for i in range(len(seen_valid)) if labels[i] == active_idx]
        active_genres = set()
        for iid in active_cluster_items:
            active_genres |= item_genres.get(iid, set())

        scores = C @ centroid
        scores = scores.copy()
        scores[seen_mask] = -2.0
        top_idx = np.lexsort((np.arange(n_items), -scores))[:M]
        recs = [item_ids[i] for i in top_idx]

        holdout_mask = seen_mask.copy()
        for iid, idx in item_index.items():
            if item_genres.get(iid, set()) & active_genres:
                holdout_mask[idx] = True
        hscores = C @ centroid
        hscores = hscores.copy()
        hscores[holdout_mask] = -2.0
        htop_idx = np.lexsort((np.arange(n_items), -hscores))[:M]
        holdout_recs = [item_ids[i] for i in htop_idx]

        n_recs = len(recs)
        n_shared = sum(1 for iid in recs if item_genres.get(iid, set()) & active_genres)
        total_recs += n_recs
        total_shared += n_shared

        holdout_r10 = recall_at_k(holdout_recs, list(test_items), k=M)
        holdout_lt = lt_recall_at_k(holdout_recs, list(test_items), lt_set, k=M)

        rows.append({
            "user_id": uid, "n_recs": n_recs, "n_shared_genre": n_shared,
            "shortcut_fraction": n_shared / n_recs if n_recs else "",
            "holdout_recall@10": holdout_r10,
            "holdout_long_tail_recall@10": "" if holdout_lt is None else holdout_lt,
        })

    pooled_fraction = total_shared / total_recs if total_recs else float("nan")
    fracs = [r["shortcut_fraction"] for r in rows if r["shortcut_fraction"] != ""]
    macro_fraction = float(np.mean(fracs)) if fracs else float("nan")
    holdout_recalls = [r["holdout_recall@10"] for r in rows]
    holdout_lts = [r["holdout_long_tail_recall@10"] for r in rows if r["holdout_long_tail_recall@10"] != ""]

    print(f"\nGENRE SHORTCUT CHECK -- Essence (K=3), MovieLens")
    print(f"Recs sharing a genre with the active cluster: pooled={pooled_fraction:.4f} ({total_shared}/{total_recs})  macro={macro_fraction:.4f}")
    print(f"Genre-holdout variant: Recall@10={np.mean(holdout_recalls):.4f}  LT-Recall@10={np.mean(holdout_lts):.4f} (n={len(holdout_lts)})")

    out_path = RESULTS_DIR / "shortcut_analysis_movielens.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_path = RESULTS_DIR / "shortcut_analysis_movielens_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["pooled_shortcut_fraction", f"{pooled_fraction:.6f}"])
        w.writerow(["macro_shortcut_fraction", f"{macro_fraction:.6f}"])
        w.writerow(["holdout_recall@10", f"{np.mean(holdout_recalls):.6f}"])
        w.writerow(["holdout_lt_recall@10", f"{np.mean(holdout_lts):.6f}"])
        w.writerow(["n_users", len(rows)])
    print(f"Saved to {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
