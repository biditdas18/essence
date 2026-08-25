"""
evaluation/step2_ablation_with_lt.py
----------------------------------------
Step 2 support: active-cluster-selection ablation (recency-select vs.
mean-select vs. Content reference), WITH LT-Recall@10 added, at the
validation-selected K per dataset. The original Step 6c ablation stages
only tracked Recall@10; this adds the LT metric needed to rebuild
Table~\\ref{tab:ablation} at the new K's.

Run:
    python evaluation/step2_ablation_with_lt.py
"""
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, ".")

RECENCY_N = 10
SEED = 42
M = 10


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


def recall_at_k(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def run_dataset(label, K, train_df, test_df, emb_meta, lt_set, item_col, out_path):
    t0 = time.time()
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)
    all_users = sorted(train_df["user_id"].unique())
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row[item_col])

    rows = []
    content_recalls, content_lts = [], []
    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows[item_col])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())
        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]

        content_recs = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M) if seen_vecs else []
        content_recalls.append(recall_at_k(content_recs, test_items))
        lt = lt_recall_at_k(content_recs, test_items, lt_set)
        if lt is not None:
            content_lts.append(lt)

        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = seen_vecs[-RECENCY_N:]
            recent_mean = np.mean(recent_vecs, axis=0)
            dists_recency = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
            centroid_recency = km.cluster_centers_[np.argmin(dists_recency)].astype(np.float32)
            full_mean = np.mean(seen_vecs, axis=0)
            dists_mean = np.linalg.norm(km.cluster_centers_ - full_mean, axis=1)
            centroid_mean = km.cluster_centers_[np.argmin(dists_mean)].astype(np.float32)
            recs_recency = top_k_unseen(normalized(centroid_recency), seen_mask, C, item_ids, M)
            recs_mean = top_k_unseen(normalized(centroid_mean), seen_mask, C, item_ids, M)
        elif seen_vecs:
            recs_recency = recs_mean = content_recs
        else:
            recs_recency = recs_mean = []

        rows.append({
            "user_id": uid,
            "recency_recall@10": recall_at_k(recs_recency, test_items),
            "recency_lt@10": lt_recall_at_k(recs_recency, test_items, lt_set),
            "meanselect_recall@10": recall_at_k(recs_mean, test_items),
            "meanselect_lt@10": lt_recall_at_k(recs_mean, test_items, lt_set),
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    recency_recall = out["recency_recall@10"].mean()
    recency_lt = out["recency_lt@10"].dropna().mean()
    mean_recall = out["meanselect_recall@10"].mean()
    mean_lt = out["meanselect_lt@10"].dropna().mean()
    content_recall = float(np.mean(content_recalls))
    content_lt = float(np.mean(content_lts)) if content_lts else float("nan")

    print(f"\n=== {label} (K={K}) === done in {time.time()-t0:.1f}s")
    print(f"  Essence (recency, r=10):   Recall@10={recency_recall:.4f}  LT-Recall@10={recency_lt:.4f}  LT/Content={recency_lt/content_lt:.2f}x")
    print(f"  Essence (mean-select):     Recall@10={mean_recall:.4f}  LT-Recall@10={mean_lt:.4f}  LT/Content={mean_lt/content_lt:.2f}x")
    print(f"  Content (reference):       Recall@10={content_recall:.4f}  LT-Recall@10={content_lt:.4f}  LT/Content=1.00x")


def main():
    # Last.fm-1K, K=10
    train_df = pd.read_pickle("data/train_interactions.pkl")
    test_df = pd.read_pickle("data/test_interactions.pkl")
    with open("embeddings/item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    with open("data/long_tail_ids.pkl", "rb") as f:
        lt_set = pickle.load(f)
    run_dataset("Last.fm-1K", 10, train_df, test_df, emb, lt_set, "track_id",
               "results/scratch_ablation_lt_lastfm_K10.csv")

    # Amazon Books, K=10
    PROC = Path("data/amazon_processed")
    train_df = pd.read_csv(PROC / "train.csv")
    test_df = pd.read_csv(PROC / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(PROC / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    lt_df = pd.read_csv(PROC / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    run_dataset("Amazon Books", 10, train_df, test_df, emb, lt_set, "item_id",
               "results/scratch_ablation_lt_amazon_K10.csv")

    # MovieLens-25M, K=15
    PROC = Path("data/movielens_processed")
    train_df = pd.read_csv(PROC / "train.csv")
    test_df = pd.read_csv(PROC / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(PROC / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    lt_df = pd.read_csv(PROC / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
    run_dataset("MovieLens-25M", 15, train_df, test_df, emb, lt_set, "item_id",
               "results/scratch_ablation_lt_movielens_K15.csv")


if __name__ == "__main__":
    main()
