"""
experiments/amazon_books/popularity_decile_recall.py
----------------------------------------------------------
Step 7: Recall@10 broken out by item popularity decile (10 buckets, from
rarest to most popular, by train-set interaction count) for Essence vs.
the three strongest baselines on Amazon (Last-Item, Avg-Last-10,
Recency-Weighted -- empirically the top performers by raw Recall@10).
Produces a popularity-recall curve rather than a single binary LT cutoff.

Decile assignment: items ranked by train-set popularity (interaction
count) and split into 10 equal-count buckets; decile 1 = rarest 10% of
items, decile 10 = most popular 10%. For each user's top-10 recs and
each decile d, we count how many of the user's test items in decile d
were recovered, divided by how many test items they had in decile d
(same per-user-then-macro-average convention as everywhere else in this
repo -- see README's averaging-convention audit).

Run:
    python experiments/amazon_books/popularity_decile_recall.py
"""

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

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
RECENCY_N = 10
RECENCY_DECAY = 0.9
SYSTEMS = ["Essence (K=3)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k):
    scores = C @ query_vec
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


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    # Popularity decile assignment (train-set interaction count; items with
    # zero train interactions -- test-only items -- go in decile 1, rarest).
    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1  # 1..10
    item_decile = dict(zip(pop_series.index, deciles))

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    per_system_decile_hits = {s: defaultdict(int) for s in SYSTEMS}
    per_system_decile_opps = {s: defaultdict(int) for s in SYSTEMS}
    per_system_decile_user_recalls = {s: defaultdict(list) for s in SYSTEMS}

    for uid in tqdm(all_users, desc="Users"):
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

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        recs = {}
        if seen_vecs:
            recs["Last-Item"] = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs["Avg-Last-10"] = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs["Recency-Weighted"] = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)

            if len(seen_vecs) >= 3:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                km.fit(np.array(seen_vecs))
                if recent_vecs:
                    recent_mean = np.mean(recent_vecs, axis=0)
                    dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                    centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                else:
                    centroid = km.cluster_centers_[0].astype(np.float32)
                recs["Essence (K=3)"] = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs["Essence (K=3)"] = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            for s in SYSTEMS:
                recs[s] = []

        # Bucket this user's test items by decile
        test_by_decile = defaultdict(set)
        for iid in test_items:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for sys_name, rec_list in recs.items():
            rec_set = set(rec_list)
            for d, items_in_decile in test_by_decile.items():
                hits = len(rec_set & items_in_decile)
                per_system_decile_hits[sys_name][d] += hits
                per_system_decile_opps[sys_name][d] += len(items_in_decile)
                per_system_decile_user_recalls[sys_name][d].append(hits / len(items_in_decile))

    rows = []
    print(f"\n{'Decile':>7} " + " ".join(f"{s:>18}" for s in SYSTEMS))
    for d in range(1, N_DECILES + 1):
        line = f"{d:>7} "
        for s in SYSTEMS:
            vals = per_system_decile_user_recalls[s].get(d, [])
            macro_recall = float(np.mean(vals)) if vals else float("nan")
            n_opp_users = len(vals)
            hits = per_system_decile_hits[s].get(d, 0)
            opps = per_system_decile_opps[s].get(d, 0)
            micro_recall = hits / opps if opps else float("nan")
            line += f"{macro_recall:>18.4f}"
            rows.append({"decile": d, "system": s, "macro_recall@10": macro_recall,
                        "micro_recall@10": micro_recall, "n_users_with_items_in_decile": n_opp_users,
                        "total_hits": hits, "total_opportunities": opps})
        print(line)

    out_path = RESULTS_DIR / "popularity_decile_recall.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
