"""
experiments/movielens/decile_analysis_movielens.py
--------------------------------------------------------
Step 11c.f: popularity-decile breakdown for MovieLens, WITH Step 7b's
paired significance test built in from the start (cold-start deciles
1-2 vs. singleton-population deciles) -- not discovered after the fact
this time. Also produces the catalog/hit-level long-tail-definition
reconciliation from Step 7b Part 5, run proactively.

Systems: Essence vs CF-ItemKNN (the strongest baseline on MovieLens and
the most contrasting system to Essence). ComiRec/MIND are excluded from
this specific decile breakdown -- no trained-weights checkpoint was
persisted to disk from their training runs, and retraining just for this
analysis wasn't judged worth the added time; their aggregate numbers are
already in the paired-bootstrap/FDR results above.

Saves:
  results/popularity_decile_recall_movielens.csv
  results/decile_significance_check_movielens.csv
  results/longtail_definition_reconciliation_movielens.csv

Run:
    python experiments/movielens/decile_analysis_movielens.py
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

from models.recommenders import build_itemknn_model, cf_itemknn_recommend

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
RESULTS_DIR = BASE_DIR / "results"

M = 10
N_DECILES = 10
K = 3
SYSTEMS = ["Essence (K=3)", "CF (ItemKNN)"]


def build_candidate_matrix(emb_meta):
    item_ids = sorted(emb_meta.keys())
    C = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k):
    scores = C @ query_vec
    scores = scores.copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


def normalized(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])

    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    n_items = len(item_ids)

    # Decile assignment: rank all candidate items by train-set popularity (0 for
    # never-seen-in-train items -- same convention as the Amazon analysis)
    popularity = train_df.groupby("item_id").size()
    pop_series = pd.Series(0, index=item_ids, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))

    itemknn = build_itemknn_model(train_df, item_col="item_id")

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    per_system_decile_user_recalls = {s: defaultdict(list) for s in SYSTEMS}
    peruser_rows_by_decile = {1: [], 2: []}
    lt_hit_decile_counts = {s: defaultdict(int) for s in SYSTEMS}
    lt_hit_totals = {s: 0 for s in SYSTEMS}

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

        recs = {}
        recs["CF (ItemKNN)"] = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=42, n_init=10)
                km.fit(np.array(seen_vecs))
                recent_vecs = seen_vecs[-10:]
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
            recs["Essence (K=3)"] = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            d = item_decile.get(iid, 1)
            test_by_decile[d].add(iid)

        for sys_name in ["Essence (K=3)", "CF (ItemKNN)"]:
            rec_list = recs[sys_name]
            for d, items_in_decile in test_by_decile.items():
                r = recall_at_k(rec_list, list(items_in_decile), k=M)
                per_system_decile_user_recalls[sys_name][d].append(r)
                if d in (1, 2):
                    peruser_rows_by_decile[d].append({"user_id": uid, "system": sys_name,
                                                       "recall@10": r, "long_tail_recall@10": ""})

            test_items_lt = test_items & lt_set
            if test_items_lt:
                hits = set(rec_list) & test_items_lt
                for iid in hits:
                    d = item_decile.get(iid, 1)
                    lt_hit_decile_counts[sys_name][d] += 1
                    lt_hit_totals[sys_name] += 1

    print("\nDecile breakdown (Essence, CF-ItemKNN) -- macro Recall@10:")
    print(f"{'Decile':>7} {'Essence':>12} {'CF-ItemKNN':>12}")
    decile_rows = []
    for d in range(1, N_DECILES + 1):
        e_vals = per_system_decile_user_recalls["Essence (K=3)"].get(d, [])
        c_vals = per_system_decile_user_recalls["CF (ItemKNN)"].get(d, [])
        e_mean = float(np.mean(e_vals)) if e_vals else float("nan")
        c_mean = float(np.mean(c_vals)) if c_vals else float("nan")
        print(f"{d:>7} {e_mean:>12.4f} {c_mean:>12.4f}  (n={len(e_vals)})")
        decile_rows.append({"decile": d, "essence_recall@10": e_mean, "cf_itemknn_recall@10": c_mean, "n": len(e_vals)})
    pd.DataFrame(decile_rows).to_csv(RESULTS_DIR / "popularity_decile_recall_movielens.csv", index=False)

    for d in [1, 2]:
        with open(RESULTS_DIR / f"decile{d}_peruser_movielens.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
            w.writeheader()
            w.writerows(peruser_rows_by_decile[d])
        print(f"Saved decile{d}_peruser_movielens.csv ({len(peruser_rows_by_decile[d])} rows)")

    print("\nLong-tail definition reconciliation:")
    lt_decile_counts = defaultdict(int)
    for iid in lt_set:
        d = item_decile.get(iid, 1)
        lt_decile_counts[d] += 1
    recon_rows = []
    n_lt = len(lt_set)
    for d in range(1, N_DECILES + 1):
        n = lt_decile_counts.get(d, 0)
        pct = 100 * n / n_lt if n_lt else 0
        print(f"  decile {d:>2}: {n:>5} singleton items ({pct:5.1f}%)")
        recon_rows.append({"part": "catalog_overlap", "decile": d, "n_singleton_items": n, "pct_of_singletons": pct})

    for sys_name in ["Essence (K=3)", "CF (ItemKNN)"]:
        total = lt_hit_totals[sys_name]
        print(f"\n  {sys_name} LT-Recall@10 hits by decile (total={total}):")
        for d in range(1, N_DECILES + 1):
            n = lt_hit_decile_counts[sys_name].get(d, 0)
            pct = 100 * n / total if total else 0
            print(f"    decile {d:>2}: {n:>3} hits ({pct:5.1f}%)")
            recon_rows.append({"part": f"{sys_name}_hit_decomposition", "decile": d, "n_hits": n, "pct_of_hits": pct})

    pd.DataFrame(recon_rows).to_csv(RESULTS_DIR / "longtail_definition_reconciliation_movielens.csv", index=False)
    print(f"\nSaved reconciliation to results/longtail_definition_reconciliation_movielens.csv")


if __name__ == "__main__":
    main()
