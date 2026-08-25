"""
evaluation/step1_ratingthresh_evaluate.py
-------------------------------------------
Tier-1 Step 1: run the 8 non-neural systems + paired bootstrap
significance + silhouette stratification on the rating>=4-filtered
Amazon Books / MovieLens-25M data built by
step1_ratingthresh_filter_build.py, at each dataset's existing
validated K (Amazon K=10, MovieLens K=15) -- this step re-tests the
SAME K, not a fresh K-selection, since the question is whether the
rating>=4 filter changes findings, not whether K should change.

MIND/ComiRec are NOT retrained under this filter (explicit scope
decision, reported): retraining two neural models from scratch for
every dataset variant tonight (rating-threshold x2, multi-seed x4,
global-cutoff x3, BPR-MF x3) is not feasible in the available time.
The 8 non-neural systems -- including Essence and all three recency
baselines -- are the ones load-bearing for the domain-dependence and
clustering-null-result claims this step is meant to test.

Outputs:
  results/scratch_ratingthresh_amazon_K10.csv       (per-user, 8 systems)
  results/scratch_ratingthresh_movielens_K15.csv    (per-user, 8 systems)
  results/scratch_ratingthresh_bootstrap_amazon_K10.csv
  results/scratch_ratingthresh_bootstrap_movielens_K15.csv
  results/scratch_ratingthresh_silhouette_amazon_K10.csv
  results/scratch_ratingthresh_silhouette_movielens_K15.csv

Run:
    python evaluation/step1_ratingthresh_evaluate.py
"""
import hashlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend

RESULTS_DIR = BASE_DIR / "results"
M = 10
SEED = 42
RECENCY_N = 10
RECENCY_DECAY = 0.9


def _stable_user_seed(uid):
    return int.from_bytes(hashlib.md5(str(uid).encode()).digest()[:4], "big")


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k=M):
    scores = C @ query_vec
    scores = scores.copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


def normalized(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def recency_weighted_query(recent_vecs, decay=RECENCY_DECAY):
    weights = np.array([decay ** i for i in range(len(recent_vecs) - 1, -1, -1)])
    weights = weights / weights.sum()
    return np.average(np.array(recent_vecs), axis=0, weights=weights)


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def run_dataset(label, proc_dir_rt4, emb_pkl_path, K, out_tag, prefix="scratch_ratingthresh"):
    t0 = time.time()
    print("=" * 60)
    print(f"[ratingthresh-eval] {label} (K={K})")
    print("=" * 60)

    train_df = pd.read_csv(proc_dir_rt4 / "train.csv")
    test_df = pd.read_csv(proc_dir_rt4 / "test.csv")
    lt_df = pd.read_csv(proc_dir_rt4 / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    with open(emb_pkl_path, "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    all_users = sorted(train_df["user_id"].unique())
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])

    print(f"  Users: {len(all_users):,}  Items in embedding cache: {n_items:,}  LT items: {len(lt_set):,}")

    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)
    itemknn = build_itemknn_model(train_df, item_col="item_id")

    ess_name = f"Essence (K={K})"
    systems = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)",
               "Last-Item", "Avg-Last-10", "Recency-Weighted", ess_name]
    rows = []
    sil = {}

    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())

        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        unseen_pool = [i for i in item_ids if i not in seen_set]
        rng_u = np.random.default_rng(_stable_user_seed(uid))
        recs_random = rng_u.choice(unseen_pool, size=min(M, len(unseen_pool)), replace=False).tolist()
        recs_pop = [iid for iid in popularity.index if iid not in seen_set][:M]
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            recs_content = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs_last_item = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs_avg_last10 = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs_recency_weighted = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)
            if len(seen_vecs) >= K:
                X = np.array(seen_vecs)
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                recs_essence = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
                if len(seen_vecs) >= K + 1 and len(set(km.labels_)) >= 2:
                    sil[uid] = silhouette_score(X, km.labels_)
            else:
                recs_essence = recs_content
        else:
            recs_content = recs_last_item = recs_avg_last10 = recs_recency_weighted = recs_essence = recs_pop

        for name, recs in zip(systems, [recs_random, recs_pop, recs_knn, recs_content,
                                        recs_last_item, recs_avg_last10, recs_recency_weighted, recs_essence]):
            rows.append({
                "user_id": uid, "system": name,
                "recall@10": recall_at_k(recs, test_items),
                "long_tail_recall@10": lt_recall_at_k(recs, test_items, lt_set),
            })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{prefix}_{out_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved per-user results -> {out_path}")

    print(f"\n  {'System':<20} {'Recall@10':>10} {'LT-Recall@10':>14}")
    for name in systems:
        sub = df[df["system"] == name]
        r10 = sub["recall@10"].mean()
        ltr = sub["long_tail_recall@10"].dropna().mean()
        print(f"  {name:<20} {r10:>10.4f} {ltr:>14.4f}")

    # Paired bootstrap: Essence vs each other system
    bootstrap_path = RESULTS_DIR / f"{prefix}_bootstrap_{out_tag}.csv"
    subprocess.run(
        [sys.executable, "evaluation/paired_bootstrap.py",
         "--input", str(out_path), "--label", f"{label} (rating>=4, K={K})",
         "--output", str(bootstrap_path)],
        cwd=str(BASE_DIR), check=True,
    )
    print(f"  Saved bootstrap results -> {bootstrap_path}")

    # Silhouette stratification (clustering-null-result check): Essence vs Recency-Weighted
    print(f"\n  Silhouette computed for {len(sil)} users")
    ess_sub = df[df["system"] == ess_name].set_index("user_id")
    rec_sub = df[df["system"] == "Recency-Weighted"].set_index("user_id")
    sil_series = pd.Series(sil)
    if len(sil_series) >= 3:
        tertiles = pd.qcut(sil_series, 3, labels=["low", "medium", "high"], duplicates="drop")
    else:
        tertiles = pd.Series(dtype=object)
    strat_rows = []
    for uid, tier in tertiles.items():
        if uid in ess_sub.index and uid in rec_sub.index:
            strat_rows.append({
                "user_id": uid, "tier": tier,
                "essence_recall@10": ess_sub.loc[uid, "recall@10"],
                "recency_recall@10": rec_sub.loc[uid, "recall@10"],
                "essence_lt@10": ess_sub.loc[uid, "long_tail_recall@10"],
                "recency_lt@10": rec_sub.loc[uid, "long_tail_recall@10"],
            })
    strat_df = pd.DataFrame(strat_rows)
    sil_path = RESULTS_DIR / f"{prefix}_silhouette_{out_tag}.csv"
    strat_df.to_csv(sil_path, index=False)
    print(f"  Saved silhouette stratification -> {sil_path}")
    if len(strat_df):
        for tier in ["low", "medium", "high"]:
            t = strat_df[strat_df["tier"] == tier]
            if len(t):
                print(f"    {tier:<8} n={len(t):<5} Essence R@10={t['essence_recall@10'].mean():.4f}  "
                      f"Recency R@10={t['recency_recall@10'].mean():.4f}")

    print(f"\n  {label} done in {time.time()-t0:.1f}s")
    return out_path, bootstrap_path, sil_path


if __name__ == "__main__":
    run_dataset(
        "Amazon Books",
        BASE_DIR / "data" / "amazon_processed_rt4",
        BASE_DIR / "data" / "amazon_processed" / "embeddings_metadata.pkl",
        K=10, out_tag="amazon_K10",
    )
    run_dataset(
        "MovieLens-25M",
        BASE_DIR / "data" / "movielens_processed_rt4",
        BASE_DIR / "data" / "movielens_processed" / "embeddings_metadata.pkl",
        K=15, out_tag="movielens_K15",
    )
