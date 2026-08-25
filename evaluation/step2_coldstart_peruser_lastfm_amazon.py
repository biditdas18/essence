"""
evaluation/step2_coldstart_peruser_lastfm_amazon.py
--------------------------------------------------------
Step 2 support: per-user decile-1/2 recall for Essence, CF-ItemKNN, and
the three recency baselines, on Last.fm-1K and Amazon Books at their
validation-selected K=10 -- needed to properly rebuild the paper's
cold-start family (6 comparisons: Essence vs CF, decile 1+2, x3 datasets)
and the Amazon-specific "loses to Last-Item/Recency-Weighted at
cold-start" paragraph, at the new K.

Run:
    python evaluation/step2_coldstart_peruser_lastfm_amazon.py
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
from models.recommenders import build_itemknn_model, cf_itemknn_recommend

K = 10
SEED = 42
M = 10
RECENCY_N = 10
RECENCY_DECAY = 0.9


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


def run_lastfm():
    t0 = time.time()
    train_df = pd.read_pickle("data/train_interactions.pkl")
    test_df = pd.read_pickle("data/test_interactions.pkl")
    with open("embeddings/item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb)
    n_items = len(item_ids)

    popularity = train_df.groupby("track_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))
    itemknn = build_itemknn_model(train_df, item_col="track_id")

    systems = [f"Essence (K={K})", "CF (ItemKNN)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]
    rows_by_decile = {1: [], 2: []}

    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["track_id"])
        seen_set = set(train_items)
        actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
        if not actual:
            continue
        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        recs = {"CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M)}
        seen_vecs = [emb[i] for i in train_items if i in emb]
        if seen_vecs:
            recs["Last-Item"] = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs["Avg-Last-10"] = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs["Recency-Weighted"] = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)
            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
                km.fit(np.array(seen_vecs))
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                recs[systems[0]] = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs[systems[0]] = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            for s in systems[2:] + [systems[0]]:
                recs[s] = []

        test_by_decile = defaultdict(set)
        for iid in actual:
            test_by_decile[item_decile.get(iid, 1)].add(iid)
        for d, target_rows in rows_by_decile.items():
            items_in_decile = test_by_decile.get(d)
            if items_in_decile:
                for s in systems:
                    target_rows.append({"user_id": uid, "system": s,
                                        "recall@10": recall_at_k(recs[s], items_in_decile)})

    for d in [1, 2]:
        pd.DataFrame(rows_by_decile[d]).to_csv(f"results/scratch_decile{d}_peruser_lastfm_K10.csv", index=False)
    print(f"[lastfm] done in {time.time()-t0:.1f}s")


def run_amazon():
    t0 = time.time()
    PROC_DIR = Path("data/amazon_processed")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb)
    n_items = len(item_ids)

    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())
    itemknn = build_itemknn_model(train_df, item_col="item_id")

    systems = [f"Essence (K={K})", "CF (ItemKNN)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]
    rows_by_decile = {1: [], 2: []}

    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())
        if not test_items:
            continue
        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        recs = {"CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M)}
        seen_vecs = [emb[i] for i in train_items if i in emb]
        if seen_vecs:
            recs["Last-Item"] = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs["Avg-Last-10"] = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs["Recency-Weighted"] = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)
            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
                km.fit(np.array(seen_vecs))
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                recs[systems[0]] = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs[systems[0]] = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            for s in systems[2:] + [systems[0]]:
                recs[s] = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            test_by_decile[item_decile.get(iid, 1)].add(iid)
        for d, target_rows in rows_by_decile.items():
            items_in_decile = test_by_decile.get(d)
            if items_in_decile:
                for s in systems:
                    target_rows.append({"user_id": uid, "system": s,
                                        "recall@10": recall_at_k(recs[s], items_in_decile)})

    for d in [1, 2]:
        pd.DataFrame(rows_by_decile[d]).to_csv(f"results/scratch_decile{d}_peruser_amazon_K10.csv", index=False)
    print(f"[amazon] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_lastfm()
    run_amazon()
