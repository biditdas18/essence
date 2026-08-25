import pickle
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, ".")
from models.recommenders import build_itemknn_model, cf_itemknn_recommend

PROC_DIR = Path("data/movielens_processed")
K = 15
SEED = 42
M = 10
N_DECILES = 10


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


def main():
    t0 = time.time()
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())
    itemknn = build_itemknn_model(train_df, item_col="item_id")

    rows_d1, rows_d2 = [], []
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
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)
        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = seen_vecs[-10:]
            recent_mean = np.mean(recent_vecs, axis=0)
            dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
            centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
            recs_ess = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
        elif seen_vecs:
            recs_ess = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            recs_ess = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            test_by_decile[item_decile.get(iid, 1)].add(iid)

        for d, target_rows in [(1, rows_d1), (2, rows_d2)]:
            items_in_decile = test_by_decile.get(d)
            if items_in_decile:
                target_rows.append({"user_id": uid, "system": f"Essence (K={K})",
                                    "recall@10": recall_at_k(recs_ess, items_in_decile)})
                target_rows.append({"user_id": uid, "system": "CF (ItemKNN)",
                                    "recall@10": recall_at_k(recs_knn, items_in_decile)})

    pd.DataFrame(rows_d1).to_csv("results/scratch_decile1_peruser_movielens_K15.csv", index=False)
    pd.DataFrame(rows_d2).to_csv("results/scratch_decile2_peruser_movielens_K15.csv", index=False)
    print(f"decile1 rows: {len(rows_d1)}, decile2 rows: {len(rows_d2)}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
