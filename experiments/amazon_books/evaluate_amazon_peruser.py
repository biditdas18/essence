"""
Phase 4 — Amazon Books full evaluation, all 5 systems, chronological split.

Systems:
  Random | Popularity | CF (ItemKNN) | Content (Avg Emb) | Essence (K=3)

Uses:
  data/amazon_processed/train.csv       (chronological split, Phase 3)
  data/amazon_processed/test.csv        (chronological split, Phase 3)
  data/amazon_processed/longtail_items.csv
  data/amazon_processed/embeddings_metadata.pkl  (Pass 1 — comparable to Last.fm)

Saves to:
  experiments/amazon_books/results_amazon_v3.csv

Note: uses Pass 1 (metadata) embeddings for Content and Essence, so this
table is directly comparable to the Last.fm-1K results_v3 table.
Pass 2 (user-review) is a separate experiment documented in RESULTS_SUMMARY.md.
"""

import csv
import hashlib
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


def _stable_user_seed(user_id) -> int:
    """PYTHONHASHSEED-independent per-user seed via MD5."""
    return int.from_bytes(
        hashlib.md5(str(user_id).encode()).digest()[:4], "big"
    )

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend

PROC_DIR = BASE_DIR / "data" / "amazon_processed"
OUT_PATH = Path(__file__).parent / "results_amazon_rerun.csv"

M = 10
K = 3


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_csv_df(path):
    return pd.read_csv(path)


# ─── Vectorized candidate matrix (metadata embeddings) ────────────────────────

def build_candidate_matrix(emb_meta: dict):
    item_ids   = sorted(emb_meta.keys())
    C          = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms      = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C         /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


def top_k_unseen(query_vec, seen_mask, C, item_ids, k=10):
    scores             = C @ query_vec
    scores[seen_mask]  = -2.0
    # Full stable sort (score desc, tie-break by index asc) — deterministic even
    # when argpartition would produce ambiguous boundary sets on tied scores.
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def recall_at_k(recs, test_items):
    if not test_items: return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)

def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test: return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 4 — Amazon Books, all 5 systems, chronological split")
    print("=" * 60)
    t_total = time.time()

    print("\n[1] Loading data ...")
    train_df = load_csv_df(PROC_DIR / "train.csv")
    test_df  = load_csv_df(PROC_DIR / "test.csv")
    lt_df    = load_csv_df(PROC_DIR / "longtail_items.csv")
    lt_set   = set(lt_df["item_id"])

    # Ensure timestamp column exists (for Essence chronological ordering)
    # Amazon train.csv has no timestamp column — Essence will fall back to index order.
    # The rows in train.csv are already in chronological order per user (Phase 3 sort).
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index  # use row order as proxy for time

    all_users = sorted(train_df["user_id"].unique())
    test_map  = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])

    print(f"  Users: {len(all_users):,}  |  LT items: {len(lt_set):,}")

    print("\n[2] Loading embeddings (metadata / Pass 1) ...")
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    print(f"  Items: {len(emb_meta):,}")

    item_ids, C, item_index = build_candidate_matrix(emb_meta)

    print("\n[3] Building popularity index ...")
    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)

    print("\n[4] Building ItemKNN model ...")
    t0 = time.time()
    itemknn = build_itemknn_model(train_df, item_col="item_id")
    print(f"  Built in {time.time()-t0:.1f}s  "
          f"(R_norm shape: {itemknn.R_norm.shape})")

    print("\n[5] Running all 5 systems ...")
    t0 = time.time()

    recall    = defaultdict(list)
    lt_recall = defaultdict(list)
    systems   = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)", "Essence (K=3)"]

    rng_global = np.random.default_rng(42)
    per_user_rows = []

    for uid in tqdm(all_users, desc="Users"):
        user_rows   = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set    = set(train_items)
        test_items  = test_map.get(uid, set())

        # Seen mask over candidate matrix
        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        # ── Random ───────────────────────────────────────────────────────────
        unseen_pool = [i for i in item_ids if i not in seen_set]
        rng_u = np.random.default_rng(_stable_user_seed(uid))
        recs_random = rng_u.choice(unseen_pool, size=min(M, len(unseen_pool)),
                                   replace=False).tolist()

        # ── Popularity ───────────────────────────────────────────────────────
        recs_pop = [iid for iid in popularity.index if iid not in seen_set][:M]

        # ── CF (ItemKNN) ──────────────────────────────────────────────────────
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)

        # ── Content (Avg Emb) ─────────────────────────────────────────────────
        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            user_vec = np.mean(seen_vecs, axis=0).astype(np.float32)
            user_vec /= (np.linalg.norm(user_vec) + 1e-8)
            recs_content = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
        else:
            recs_content = recs_pop

        # ── Essence (K=3) ─────────────────────────────────────────────────────
        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = [emb_meta[i] for i in train_items[-10:] if i in emb_meta]
            if recent_vecs:
                recent_mean = np.mean(recent_vecs, axis=0)
                dists       = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid    = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
            else:
                centroid = km.cluster_centers_[0].astype(np.float32)
            centroid /= (np.linalg.norm(centroid) + 1e-8)
            recs_essence = top_k_unseen(centroid, seen_mask, C, item_ids, M)
        else:
            recs_essence = recs_content

        # ── Accumulate metrics ────────────────────────────────────────────────
        for name, recs in zip(systems, [recs_random, recs_pop, recs_knn,
                                        recs_content, recs_essence]):
            r10v = recall_at_k(recs, test_items)
            recall[name].append(r10v)
            ltr = lt_recall_at_k(recs, test_items, lt_set)
            if ltr is not None:
                lt_recall[name].append(ltr)
            per_user_rows.append({"user_id": uid, "system": name,
                                  "recall@10": r10v,
                                  "long_tail_recall@10": ("" if ltr is None else ltr)})

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Compute summary ───────────────────────────────────────────────────────
    random_r10 = float(np.mean(recall["Random"]))
    content_lt = float(np.mean(lt_recall["Content (Avg Emb)"])) \
                 if lt_recall["Content (Avg Emb)"] else 0.0

    rows_out = []
    print("\n" + "=" * 72)
    print("RESULTS — Amazon Books, Phase 4 (chronological split, Pass 1 emb)")
    print("=" * 72)
    print(f"  {'System':<20} {'R@10':>7} {'Lift':>7} {'LT-R@10':>9} {'LT/Ctn':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
    for name in systems:
        r10  = float(np.mean(recall[name]))
        ltr  = float(np.mean(lt_recall[name])) if lt_recall[name] else 0.0
        lift = r10 / random_r10 if random_r10 > 0 else float("nan")
        lt_c = ltr / content_lt if content_lt > 0 else float("nan")
        lt_c_str = f"{lt_c:.3f}x" if not np.isnan(lt_c) else "  —"
        print(f"  {name:<20} {r10:>7.4f} {lift:>6.0f}x {ltr:>9.4f} {lt_c_str:>8}")
        rows_out.append({
            "dataset": "Amazon Books",
            "system": name,
            "Recall@10": f"{r10:.4f}",
            "LT-Recall@10": f"{ltr:.4f}",
            "Lift_vs_random": f"{lift:.1f}",
            "LT_content_ratio": f"{lt_c:.3f}" if not np.isnan(lt_c) else "nan",
        })
    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  LT users (have ≥1 singleton test item): "
          f"{len(lt_recall['Content (Avg Emb)']):,} / {len(all_users):,}")

    # Save aggregate summary
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  Saved aggregate to {OUT_PATH}")

    # Save per-user rows
    peruser_path = Path(__file__).parent / "results_amazon_peruser.csv"
    with open(peruser_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
        w.writeheader()
        w.writerows(per_user_rows)
    print(f"  Saved per-user to {peruser_path}")
    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
