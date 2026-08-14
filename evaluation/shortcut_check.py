"""
evaluation/shortcut_check.py
------------------------------
Checks whether Essence's gains are explained by a trivial artist/track
shortcut: does it just recommend more items by artists already in the
user's active cluster, rather than genuinely different (but semantically
related) content?

Two analyses, both over all users:

  1. Shortcut fraction: for each user's Essence top-10 recommendations,
     what fraction share an artist with an item in the user's active
     K-means cluster (the cluster whose centroid produced those recs)?

  2. Artist-holdout variant: rerun Essence excluding any candidate item
     whose artist appears in the active cluster, report Recall@10 /
     LT-Recall@10 for this variant (compare against the unrestricted
     Essence numbers to see how much of Essence's advantage the artist
     shortcut accounts for).

Saves per-user rows for both analyses to results/shortcut_analysis.csv.

Run:
    python evaluation/shortcut_check.py
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import cosine_similarity, content_recommend
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

K = 3
M = 10
SEED = 42


def load_data():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as fh:
        item_embedding_map = pickle.load(fh)
    with open(DATA_DIR / "long_tail_ids.pkl", "rb") as fh:
        long_tail_ids = pickle.load(fh)

    # track_id -> artist_id, built from train ∪ test (matches item_embedding_map's coverage)
    all_rows = pd.concat([train_df, test_df])[["track_id", "artist_id"]].drop_duplicates("track_id")
    item_artist_map = dict(zip(all_rows["track_id"], all_rows["artist_id"]))

    return train_df, test_df, item_embedding_map, long_tail_ids, item_artist_map


def essence_cluster_analysis(user_id, train_df, item_embedding_map, item_artist_map,
                             K=K, M=M, seed=SEED):
    """
    Reimplements essence_recommend's clustering, but additionally returns
    the set of artists present in the active cluster and an artist-holdout
    variant of the recommendation list.

    Returns
    -------
    recs            : normal Essence top-M recs
    holdout_recs     : top-M recs excluding any candidate sharing an artist
                       with the active cluster
    active_artists   : set of artist_ids in the active cluster
    """
    user_rows = train_df[train_df["user_id"] == user_id].sort_values("timestamp")
    seen = list(user_rows["track_id"])
    seen_valid = [i for i in seen if i in item_embedding_map]
    vecs = [item_embedding_map[i] for i in seen_valid]

    if len(vecs) < K:
        # Fallback: no clustering possible; treat the whole (small) history as
        # the "active cluster" so the shortcut check still has a well-defined
        # artist set to compare against.
        recs = content_recommend(user_id, train_df, item_embedding_map, M)
        active_artists = {item_artist_map[i] for i in seen_valid if i in item_artist_map}
        seen_set = set(seen)
        candidates = {
            tid: emb for tid, emb in item_embedding_map.items()
            if tid not in seen_set and item_artist_map.get(tid) not in active_artists
        }
        scores = {tid: cosine_similarity(np.mean(vecs, axis=0) if vecs else np.zeros(1), emb)
                 for tid, emb in candidates.items()} if vecs else {}
        holdout_recs = sorted(scores, key=scores.get, reverse=True)[:M] if scores else []
        return recs, holdout_recs, active_artists

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(np.array(vecs))

    recent = seen[-10:]
    recent_vecs = [item_embedding_map[i] for i in recent if i in item_embedding_map]
    if recent_vecs:
        recent_mean = np.mean(recent_vecs, axis=0)
        dists = [np.linalg.norm(c - recent_mean) for c in km.cluster_centers_]
        active_idx = int(np.argmin(dists))
    else:
        active_idx = 0
    active_centroid = km.cluster_centers_[active_idx]

    labels = km.labels_
    active_cluster_items = [seen_valid[i] for i in range(len(seen_valid)) if labels[i] == active_idx]
    active_artists = {item_artist_map[i] for i in active_cluster_items if i in item_artist_map}

    seen_set = set(seen)

    # Normal recs
    candidates = {tid: emb for tid, emb in item_embedding_map.items() if tid not in seen_set}
    scores = {tid: cosine_similarity(active_centroid, emb) for tid, emb in candidates.items()}
    recs = sorted(scores, key=scores.get, reverse=True)[:M]

    # Artist-holdout recs: exclude candidates sharing an artist with the active cluster
    holdout_candidates = {
        tid: emb for tid, emb in candidates.items()
        if item_artist_map.get(tid) not in active_artists
    }
    holdout_scores = {tid: cosine_similarity(active_centroid, emb) for tid, emb in holdout_candidates.items()}
    holdout_recs = sorted(holdout_scores, key=holdout_scores.get, reverse=True)[:M]

    return recs, holdout_recs, active_artists


def main():
    train_df, test_df, item_embedding_map, long_tail_ids, item_artist_map = load_data()
    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))
    print(f"[shortcut_check] Evaluating {len(all_users)} users (K={K}, M={M})\n")

    rows = []
    total_recs = 0
    total_shared = 0

    for user_id in tqdm(all_users, desc="Users"):
        actual = test_df[test_df["user_id"] == user_id]["track_id"].tolist()

        recs, holdout_recs, active_artists = essence_cluster_analysis(
            user_id, train_df, item_embedding_map, item_artist_map
        )

        n_recs = len(recs)
        n_shared = sum(1 for tid in recs if item_artist_map.get(tid) in active_artists)
        total_recs += n_recs
        total_shared += n_shared

        holdout_r10 = recall_at_k(holdout_recs, actual, k=M)
        holdout_lt = long_tail_recall_at_k(holdout_recs, actual, long_tail_ids, k=M)

        rows.append({
            "user_id": user_id,
            "n_recs": n_recs,
            "n_shared_artist": n_shared,
            "shortcut_fraction": n_shared / n_recs if n_recs else "",
            "holdout_recall@10": holdout_r10,
            "holdout_long_tail_recall@10": "" if holdout_lt is None else holdout_lt,
        })

    pooled_fraction = total_shared / total_recs if total_recs else float("nan")
    macro_fraction = float(np.mean([r["shortcut_fraction"] for r in rows if r["shortcut_fraction"] != ""]))

    holdout_recalls = [r["holdout_recall@10"] for r in rows]
    holdout_lt = [r["holdout_long_tail_recall@10"] for r in rows if r["holdout_long_tail_recall@10"] != ""]

    print(f"\n{'='*60}")
    print("  ARTIST/TRACK SHORTCUT CHECK — Essence (K=3)")
    print(f"{'='*60}")
    print(f"  Recs sharing an artist with the active cluster:")
    print(f"    pooled fraction (Σshared / Σrecs): {pooled_fraction:.4f}  ({total_shared}/{total_recs})")
    print(f"    macro fraction  (mean of per-user):  {macro_fraction:.4f}")
    print(f"\n  Artist-holdout variant (same-artist candidates excluded):")
    print(f"    Recall@10:    {np.mean(holdout_recalls):.4f}")
    print(f"    LT-Recall@10: {np.mean(holdout_lt):.4f}  (n={len(holdout_lt)})")
    print()

    out_path = RESULTS_DIR / "shortcut_analysis.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[shortcut_check] Saved per-user rows to {out_path}")

    summary_path = RESULTS_DIR / "shortcut_analysis_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["pooled_shortcut_fraction", f"{pooled_fraction:.6f}"])
        w.writerow(["macro_shortcut_fraction", f"{macro_fraction:.6f}"])
        w.writerow(["holdout_recall@10", f"{np.mean(holdout_recalls):.6f}"])
        w.writerow(["holdout_lt_recall@10", f"{np.mean(holdout_lt):.6f}"])
        w.writerow(["n_users", len(all_users)])
    print(f"[shortcut_check] Saved summary to {summary_path}\n")


if __name__ == "__main__":
    main()
