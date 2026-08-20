"""
experiments/amazon_books/longtail_definition_reconciliation.py
--------------------------------------------------------------------
Step 7b part 5: where do the singleton (binary) long-tail definition and
the popularity-decile definition actually disagree?

Two things reported, factually, whatever they show:

  1. Catalog-level overlap: what fraction of singleton items (train
     count == 1, the standard LT definition) fall in decile 1, decile 2,
     or deciles 3+? (Decile boundaries are equal-ITEM-count buckets via
     pd.qcut on rank; with 42,878 singletons out of 61,727 catalog items
     -- 69.5% of the catalog -- and only ~6,173 items per decile,
     singletons cannot possibly all fit in decile 1. Tie-breaking among
     the many tied-at-1-interaction items, via rank(method="first"),
     smears them across multiple deciles based on incidental row order,
     not any real popularity difference between them.)

  2. Hit-level decomposition: of Essence's actual LT-Recall@10 hits (top-10
     recs that land on a user's singleton test items -- the numerator of
     the headline metric), what fraction fall in decile 1, decile 2, vs.
     deciles 3+? This shows how much of the headline LT-Recall win is
     actually coming from the same items Step 7's decile-1/2 breakdown
     was testing, vs. singleton items sitting elsewhere in the decile
     range.

Run:
    python experiments/amazon_books/longtail_definition_reconciliation.py
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


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])
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

    # --- Part 1: catalog-level overlap ---------------------------------------
    print("=" * 70)
    print("PART 1: catalog-level overlap (singleton items x popularity decile)")
    print("=" * 70)
    lt_decile_counts = defaultdict(int)
    for iid in lt_set:
        d = item_decile.get(iid, 1)
        lt_decile_counts[d] += 1

    n_lt = len(lt_set)
    catalog_rows = []
    for d in range(1, N_DECILES + 1):
        n_in_decile = lt_decile_counts.get(d, 0)
        pct = 100 * n_in_decile / n_lt if n_lt else 0
        print(f"  decile {d:>2}: {n_in_decile:>6,} singleton items ({pct:5.1f}% of all {n_lt:,} singletons)")
        catalog_rows.append({"part": "catalog_overlap", "decile": d,
                            "n_singleton_items_in_decile": n_in_decile,
                            "pct_of_all_singletons": pct})

    # --- Part 2: hit-level decomposition (Essence's actual LT-Recall@10 hits) ---
    print("\n" + "=" * 70)
    print("PART 2: decile breakdown of Essence's actual LT-Recall@10 hits")
    print("=" * 70)

    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    hit_decile_counts = defaultdict(int)
    total_hits = 0

    for uid in tqdm(all_users, desc="Users"):
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())
        test_items_lt = {i for i in test_items if i in lt_set}
        if not test_items_lt:
            continue

        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if not seen_vecs:
            continue

        if len(seen_vecs) >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = seen_vecs[-10:]
            if recent_vecs:
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
            else:
                centroid = km.cluster_centers_[0].astype(np.float32)
            recs = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
        else:
            recs = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)

        hits = set(recs) & test_items_lt
        for iid in hits:
            d = item_decile.get(iid, 1)
            hit_decile_counts[d] += 1
            total_hits += 1

    hit_rows = []
    for d in range(1, N_DECILES + 1):
        n_hits = hit_decile_counts.get(d, 0)
        pct = 100 * n_hits / total_hits if total_hits else 0
        print(f"  decile {d:>2}: {n_hits:>4} hits ({pct:5.1f}% of all {total_hits} LT-Recall@10 hits)")
        hit_rows.append({"part": "hit_decomposition", "decile": d,
                        "n_lt_recall_hits_in_decile": n_hits,
                        "pct_of_all_lt_hits": pct})

    decile12_pct = sum(r["pct_of_all_lt_hits"] for r in hit_rows if r["decile"] in (1, 2))
    print(f"\n  Deciles 1+2 combined account for {decile12_pct:.1f}% of Essence's LT-Recall@10 hits")
    print(f"  (i.e. {100-decile12_pct:.1f}% of the headline LT-Recall win comes from singleton items")
    print(f"   sitting OUTSIDE deciles 1-2, where Step 7's targeted test doesn't cover)")

    out_path = RESULTS_DIR / "longtail_definition_reconciliation.csv"
    pd.DataFrame(catalog_rows + hit_rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
