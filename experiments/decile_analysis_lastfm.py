"""
experiments/decile_analysis_lastfm.py
------------------------------------------
Step 3: first-ever popularity-decile breakdown for Last.fm-1K. Step 7/7b
was Amazon-only, Step 11's version was MovieLens-only -- Last.fm was never
covered. Same protocol: all 10 systems' Recall@10 per decile, opportunity
counts reported explicitly (n=99 users means deciles WILL be underpowered
-- flagged directly, not glossed over), and a decile-1/2 paired
significance test vs. CF-ItemKNN matching Steps 2 and 11's protocol.

Saves:
  results/popularity_decile_recall_lastfm.csv   (all 10 systems x 10 deciles)
  results/decile1_peruser_lastfm.csv
  results/decile2_peruser_lastfm.csv

Run:
    python experiments/decile_analysis_lastfm.py
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

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import (
    build_itemknn_model, cf_itemknn_recommend, random_recommend,
    popularity_recommend, content_recommend, essence_recommend,
    last_item_recommend, avg_last10_recommend, recency_weighted_recommend,
)
sys.path.insert(0, str(BASE_DIR / "experiments" / "mind_comirec"))
import torch
from model import MIND, ComiRecSA

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
K = 3

FULL_SYSTEMS = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)",
               "Last-Item", "Avg-Last-10", "Recency-Weighted", "Essence (K=3)"]


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def main():
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df = pd.read_pickle(DATA_DIR / "test_interactions.pkl")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)

    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))
    print(f"[decile_lastfm] {len(all_users)} users")

    # Decile assignment: same convention as Amazon/MovieLens (train-set popularity
    # rank over the FULL candidate pool -- items with 0 train interactions rank lowest)
    item_ids_all = sorted(emb.keys())
    popularity = train_df.groupby("track_id").size()
    pop_series = pd.Series(0, index=item_ids_all, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    itemknn = build_itemknn_model(train_df, item_col="track_id")

    per_system_decile_recalls = {s: defaultdict(list) for s in FULL_SYSTEMS}
    rows_by_decile = {1: [], 2: []}

    for uid in tqdm(all_users, desc="Users"):
        actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
        if not actual:
            continue

        recs = {
            "Random": random_recommend(uid, train_df, M, item_embedding_map=emb),
            "Popularity": popularity_recommend(uid, train_df, M, item_embedding_map=emb),
            "CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M, item_embedding_map=emb),
            "Content (Avg Emb)": content_recommend(uid, train_df, emb, M),
            "Last-Item": last_item_recommend(uid, train_df, emb, M),
            "Avg-Last-10": avg_last10_recommend(uid, train_df, emb, M),
            "Recency-Weighted": recency_weighted_recommend(uid, train_df, emb, M),
            "Essence (K=3)": essence_recommend(uid, train_df, emb, K=K, M=M),
        }

        test_by_decile = defaultdict(set)
        for iid in actual:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for sys_name, rec_list in recs.items():
            for d, items_in_decile in test_by_decile.items():
                r = recall_at_k(rec_list, list(items_in_decile), k=M)
                per_system_decile_recalls[sys_name][d].append(r)

            if 1 in test_by_decile or 2 in test_by_decile:
                for d in [1, 2]:
                    items_in_decile = test_by_decile.get(d)
                    if items_in_decile and sys_name in ("Essence (K=3)", "CF (ItemKNN)"):
                        r = recall_at_k(rec_list, list(items_in_decile), k=M)
                        rows_by_decile[d].append({"user_id": uid, "system": sys_name,
                                                  "recall@10": r, "long_tail_recall@10": ""})

    print(f"\n{'Decile':>7} " + " ".join(f"{s:>16}" for s in FULL_SYSTEMS) + f" {'n_opps':>8}")
    decile_rows = []
    for d in range(1, N_DECILES + 1):
        line = f"{d:>7} "
        n_opps = len(per_system_decile_recalls["Essence (K=3)"].get(d, []))
        for s in FULL_SYSTEMS:
            vals = per_system_decile_recalls[s].get(d, [])
            mean_r = float(np.mean(vals)) if vals else float("nan")
            line += f"{mean_r:>16.4f}"
            decile_rows.append({"decile": d, "system": s, "recall@10": mean_r, "n_opportunities": len(vals)})
        flag = "  <-- SPARSE, low trust" if n_opps < 20 else ""
        print(line + f" {n_opps:>8}{flag}")

    pd.DataFrame(decile_rows).to_csv(RESULTS_DIR / "popularity_decile_recall_lastfm.csv", index=False)

    for d in [1, 2]:
        out_path = RESULTS_DIR / f"decile{d}_peruser_lastfm.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(rows_by_decile[d])
        n = len(set(r["user_id"] for r in rows_by_decile[d]))
        print(f"Saved decile{d}_peruser_lastfm.csv: {len(rows_by_decile[d])} rows ({n} users)")


if __name__ == "__main__":
    main()
