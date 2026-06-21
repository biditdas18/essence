"""
Pass 2 — Amazon Books evaluation using user-review embeddings for profile building.

Systems:
  Random | Popularity | CF (ItemKNN) | Content (Avg Emb) | Essence (K=3)

Differences from Pass 1 (evaluate_amazon_peruser.py):
  - Loads embeddings_user_review.pkl keyed by (user_id, item_id)
  - Content and Essence build profile vectors from review embeddings
  - Candidate matrix C is still built from metadata embeddings (Pass 1 space)
  - Random, Popularity, CF are IDENTICAL to Pass 1

Saves to:
  experiments/amazon_books/results_amazon_pass2_rerun.csv   (aggregate)
  experiments/amazon_books/results_amazon_pass2_peruser.csv (per-user)
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
OUT_PATH = Path(__file__).parent / "results_amazon_pass2_rerun.csv"

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
    # Full stable sort (score desc, tie-break by index asc) — deterministic.
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
    print("PASS 2 — Amazon Books, review-embedding profiles, metadata candidates")
    print("=" * 60)
    t_total = time.time()

    print("\n[1] Loading data ...")
    train_df = load_csv_df(PROC_DIR / "train.csv")
    test_df  = load_csv_df(PROC_DIR / "test.csv")
    lt_df    = load_csv_df(PROC_DIR / "longtail_items.csv")
    lt_set   = set(lt_df["item_id"])

    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index

    all_users = sorted(train_df["user_id"].unique())
    test_map  = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])

    print(f"  Users: {len(all_users):,}  |  LT items: {len(lt_set):,}")

    print("\n[2] Loading metadata embeddings (candidate matrix, Pass 1) ...")
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    print(f"  Items in metadata pkl: {len(emb_meta):,}")

    item_ids, C, item_index = build_candidate_matrix(emb_meta)

    print("\n[3] Loading user-review embeddings (profile vectors, Pass 2) ...")
    with open(PROC_DIR / "embeddings_user_review.pkl", "rb") as f:
        emb_review = pickle.load(f)
    sample_key = next(iter(emb_review))
    print(f"  Entries in review pkl: {len(emb_review):,}  |  sample key: {sample_key!r}")

    print("\n[4] Building popularity index ...")
    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)

    print("\n[5] Building ItemKNN model ...")
    t0 = time.time()
    itemknn = build_itemknn_model(train_df, item_col="item_id")
    print(f"  Built in {time.time()-t0:.1f}s  "
          f"(R_norm shape: {itemknn.R_norm.shape})")

    print("\n[6] Running all 5 systems ...")
    t0 = time.time()

    recall    = defaultdict(list)
    lt_recall = defaultdict(list)
    systems   = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)", "Essence (K=3)"]

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

        # ── Random (identical to Pass 1) ─────────────────────────────────────
        unseen_pool = [i for i in item_ids if i not in seen_set]
        rng_u = np.random.default_rng(_stable_user_seed(uid))
        recs_random = rng_u.choice(unseen_pool, size=min(M, len(unseen_pool)),
                                   replace=False).tolist()

        # ── Popularity (identical to Pass 1) ─────────────────────────────────
        recs_pop = [iid for iid in popularity.index if iid not in seen_set][:M]

        # ── CF (ItemKNN) (identical to Pass 1 — embedding-independent) ───────
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)

        # ── Content (Avg Emb) — profile from review embeddings ────────────────
        seen_vecs = [emb_review[(uid, i)] for i in train_items if (uid, i) in emb_review]
        if seen_vecs:
            user_vec = np.mean(seen_vecs, axis=0).astype(np.float32)
            user_vec /= (np.linalg.norm(user_vec) + 1e-8)
            recs_content = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
        else:
            recs_content = recs_pop

        # ── Essence (K=3) — profile from review embeddings ───────────────────
        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = [emb_review[(uid, i)] for i in train_items[-10:] if (uid, i) in emb_review]
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
    print("RESULTS — Amazon Books, Pass 2 (review-embedding profiles)")
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
            "dataset": "Amazon Books Pass 2",
            "system": name,
            "Recall@10": f"{r10:.4f}",
            "LT-Recall@10": f"{ltr:.4f}",
            "Lift_vs_random": f"{lift:.1f}",
            "LT_content_ratio": f"{lt_c:.3f}" if not np.isnan(lt_c) else "nan",
        })
    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  LT users (have ≥1 singleton test item): "
          f"{len(lt_recall['Content (Avg Emb)']):,} / {len(all_users):,}")

    # ── Assertions / comparison ───────────────────────────────────────────────
    reference = {
        "CF (ItemKNN)":      (0.0031, 0.0137),
        "Content (Avg Emb)": (0.0027, 0.0012),
        "Essence (K=3)":     (0.0041, 0.0034),
    }

    cf_r10  = float(np.mean(recall["CF (ItemKNN)"]))
    cf_ltr  = float(np.mean(lt_recall["CF (ItemKNN)"])) if lt_recall["CF (ItemKNN)"] else 0.0
    ref_cf_r10, ref_cf_ltr = reference["CF (ItemKNN)"]

    print("\n" + "=" * 72)
    print("ASSERTIONS / COMPARISON vs paper Table 6 Pass-2 reference")
    print("=" * 72)
    cf_r10_diff = abs(cf_r10 - ref_cf_r10)
    cf_ltr_diff = abs(cf_ltr - ref_cf_ltr)
    cf_ok = cf_r10_diff <= 0.0002 and cf_ltr_diff <= 0.0002
    print(f"\n  [HARD ASSERT] CF (ItemKNN) — embedding-independent, must match Pass-1:")
    print(f"    Recall@10:     recomputed={cf_r10:.4f}  ref={ref_cf_r10:.4f}  diff={cf_r10_diff:.4f}  {'PASS' if cf_r10_diff <= 0.0002 else 'FAIL'}")
    print(f"    LT-Recall@10:  recomputed={cf_ltr:.4f}  ref={ref_cf_ltr:.4f}  diff={cf_ltr_diff:.4f}  {'PASS' if cf_ltr_diff <= 0.0002 else 'FAIL'}")
    print(f"    Overall: {'PASS — CF matches Pass-1 within tolerance' if cf_ok else 'FAIL — pipeline error, STOPPING'}")

    if not cf_ok:
        print("\n  STOPPING: CF assert failed. Do not interpret Content/Essence results.")
        return

    print(f"\n  [INFO] Content (Avg Emb) vs reference (no assert):")
    for label, key in [("Recall@10", "recall"), ("LT-Recall@10", "lt_recall")]:
        recomp = float(np.mean(recall["Content (Avg Emb)"])) if key == "recall" \
                 else (float(np.mean(lt_recall["Content (Avg Emb)"])) if lt_recall["Content (Avg Emb)"] else 0.0)
        ref_v  = reference["Content (Avg Emb)"][0 if key == "recall" else 1]
        print(f"    {label:<14} recomputed={recomp:.4f}  ref={ref_v:.4f}  diff={abs(recomp-ref_v):.4f}")

    print(f"\n  [INFO] Essence (K=3) vs reference (no assert):")
    for label, key in [("Recall@10", "recall"), ("LT-Recall@10", "lt_recall")]:
        recomp = float(np.mean(recall["Essence (K=3)"])) if key == "recall" \
                 else (float(np.mean(lt_recall["Essence (K=3)"])) if lt_recall["Essence (K=3)"] else 0.0)
        ref_v  = reference["Essence (K=3)"][0 if key == "recall" else 1]
        print(f"    {label:<14} recomputed={recomp:.4f}  ref={ref_v:.4f}  diff={abs(recomp-ref_v):.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  Saved aggregate to {OUT_PATH}")

    peruser_path = Path(__file__).parent / "results_amazon_pass2_peruser.csv"
    with open(peruser_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
        w.writeheader()
        w.writerows(per_user_rows)
    print(f"  Saved per-user to {peruser_path}")
    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
