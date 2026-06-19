"""
Step 4 — Evaluate Amazon Books recommenders (vectorized).

Candidate scoring uses pre-built numpy matrix (N_items × 384) so
all 61K cosine similarities per user are one matrix multiply, not
a Python loop.

Metrics:
  Recall@10     : |R_u ∩ Test_u| / |Test_u|, averaged over all users
  LT-Recall@10  : same but restricted to singleton test items;
                  users with no singleton test items are excluded

Produces:
  TABLE A — metadata embeddings (Pass 1)
  TABLE B — user-review enriched embeddings (Pass 2)
  Saved to: experiments/amazon_books/results_amazon.csv
"""

import csv
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon_processed"
OUT_DIR  = Path(__file__).resolve().parent

M = 10  # recommendation list length


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ─── Vectorized candidate matrix ──────────────────────────────────────────────

def build_candidate_matrix(emb_meta: dict):
    """
    Returns:
      item_ids  : list of item_id strings (length N)
      C         : np.ndarray (N × 384), L2-normalised rows
      item_index: dict {item_id: row index}
    """
    item_ids  = sorted(emb_meta.keys())
    C         = np.array([emb_meta[i] for i in item_ids], dtype=np.float32)
    norms     = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    C        /= norms
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    return item_ids, C, item_index


# ─── Profile vector helpers ───────────────────────────────────────────────────

def get_profile_vecs(user_id, item_list, profile_emb: dict, use_pair: bool) -> np.ndarray:
    vecs = []
    for iid in item_list:
        key = (user_id, iid) if use_pair else iid
        v   = profile_emb.get(key)
        if v is not None:
            vecs.append(v)
    return np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 384), dtype=np.float32)


def top_k_unseen(query_vec: np.ndarray, seen_mask: np.ndarray,
                 C: np.ndarray, item_ids: list, k: int) -> list:
    """
    query_vec  : (384,) already L2-normalised
    seen_mask  : bool array (N,), True where item is in user's train set
    C          : (N × 384) L2-normalised candidate matrix
    Returns top-k item_ids among unseen items.
    """
    scores              = C @ query_vec          # (N,)
    scores[seen_mask]   = -2.0                   # mask out seen items
    # Full stable sort (score desc, tie-break by index asc) — deterministic even
    # when argpartition would produce ambiguous boundary sets on tied scores.
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


# ─── Popularity (CF) precompute ───────────────────────────────────────────────

def build_popularity(train_df) -> pd.Series:
    return train_df.groupby("item_id").size().sort_values(ascending=False)


def cf_top_k(popularity: pd.Series, seen_set: set, k: int) -> list:
    return [iid for iid in popularity.index if iid not in seen_set][:k]


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(label, train_df, test_map, lt_test_map, all_users,
                   profile_emb, item_ids, C, item_index, popularity,
                   progress_every=500):

    use_pair = isinstance(next(iter(profile_emb)), tuple)
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  Profile key type: {'(user_id, item_id)' if use_pair else 'item_id'}")
    print(f"{'='*58}")
    t0 = time.time()

    recall    = {"CF": [], "Content": [], "Essence": []}
    lt_recall = {"CF": [], "Content": [], "Essence": []}

    for i, uid in enumerate(all_users):
        if i > 0 and i % progress_every == 0:
            print(f"  ... {i}/{len(all_users)} users  ({time.time()-t0:.0f}s elapsed)")

        user_rows   = train_df[train_df["user_id"] == uid].sort_values("timestamp", na_position="last")
        train_items = list(user_rows["item_id"])
        seen_set    = set(train_items)

        test_items    = test_map.get(uid, set())
        lt_test_items = lt_test_map.get(uid, set())

        # Seen mask over candidate matrix
        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        # ── CF ──────────────────────────────────────────────────────────────
        recs_cf = cf_top_k(popularity, seen_set, M)
        recall["CF"].append(_recall(recs_cf, test_items))
        if lt_test_items:
            lt_recall["CF"].append(_recall(recs_cf, lt_test_items))

        # ── Content ─────────────────────────────────────────────────────────
        vecs = get_profile_vecs(uid, train_items, profile_emb, use_pair)
        if len(vecs) > 0:
            user_vec = vecs.mean(axis=0)
            norm     = np.linalg.norm(user_vec) + 1e-8
            user_vec /= norm
            recs_content = top_k_unseen(user_vec, seen_mask, C, item_ids, M)
        else:
            recs_content = []
        recall["Content"].append(_recall(recs_content, test_items))
        if lt_test_items:
            lt_recall["Content"].append(_recall(recs_content, lt_test_items))

        # ── Essence ─────────────────────────────────────────────────────────
        if len(vecs) >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            km.fit(vecs)
            # Active centroid: closest to mean of last-10 items
            recent_vecs = get_profile_vecs(uid, train_items[-10:], profile_emb, use_pair)
            if len(recent_vecs) > 0:
                recent_mean = recent_vecs.mean(axis=0)
                dists       = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid    = km.cluster_centers_[np.argmin(dists)]
            else:
                centroid    = km.cluster_centers_[0]
            norm     = np.linalg.norm(centroid) + 1e-8
            centroid /= norm
            recs_essence = top_k_unseen(centroid, seen_mask, C, item_ids, M)
        else:
            recs_essence = recs_content   # fallback
        recall["Essence"].append(_recall(recs_essence, test_items))
        if lt_test_items:
            lt_recall["Essence"].append(_recall(recs_essence, lt_test_items))

    elapsed = time.time() - t0
    print(f"  Finished {len(all_users)} users in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    content_lt = float(np.mean(lt_recall["Content"])) if lt_recall["Content"] else 0.0
    results = {}
    for sys in ["CF", "Content", "Essence"]:
        r   = float(np.mean(recall[sys]))
        ltr = float(np.mean(lt_recall[sys])) if lt_recall[sys] else 0.0
        ratio = ltr / content_lt if content_lt > 0 else float("nan")
        results[sys] = {"Recall@10": r, "LT-Recall@10": ltr, "LT/Content Ratio": ratio}
    return results


def _recall(recs, test_items):
    if not test_items:
        return 0.0
    return sum(1 for r in recs if r in test_items) / len(test_items)


# ─── Output helpers ───────────────────────────────────────────────────────────

def print_table(label, results):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'System':<16} {'Recall@10':>10} {'LT-Recall@10':>13} {'LT/Content Ratio':>17}")
    print(f"  {'-'*16} {'-'*10} {'-'*13} {'-'*17}")
    for sys in ["CF", "Content", "Essence"]:
        r = results[sys]
        ratio = f"{r['LT/Content Ratio']:.3f}x" if not np.isnan(r["LT/Content Ratio"]) else "    —"
        print(f"  {sys:<16} {r['Recall@10']:>10.4f} {r['LT-Recall@10']:>13.4f} {ratio:>17}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading train/test data ...")
    train_rows = load_csv(PROC_DIR / "train.csv")
    test_rows  = load_csv(PROC_DIR / "test.csv")
    lt_rows    = load_csv(PROC_DIR / "longtail_items.csv")

    longtail_set = {r["item_id"] for r in lt_rows}
    train_df     = pd.DataFrame(train_rows)
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = None

    all_users = sorted(train_df["user_id"].unique())

    test_map = defaultdict(set)
    for r in test_rows:
        test_map[r["user_id"]].add(r["item_id"])

    lt_test_map = defaultdict(set)
    for r in test_rows:
        if r["item_id"] in longtail_set:
            lt_test_map[r["user_id"]].add(r["item_id"])

    users_with_lt = sum(1 for uid in all_users if lt_test_map.get(uid))
    print(f"  Users: {len(all_users):,}  |  LT items: {len(longtail_set):,}  |  Users w/ LT test: {users_with_lt:,}")

    print("\nLoading embeddings ...")
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    with open(PROC_DIR / "embeddings_user_review.pkl", "rb") as f:
        emb_review = pickle.load(f)
    print(f"  Metadata embeddings    : {len(emb_meta):,} items")
    print(f"  User-review embeddings : {len(emb_review):,} pairs")

    print("\nBuilding candidate matrix (61K × 384) ...")
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    print(f"  Matrix shape: {C.shape}  dtype: {C.dtype}")

    popularity = build_popularity(train_df)

    # ── RUN A: metadata profile + metadata candidates ─────────────────────────
    results_a = run_evaluation(
        "TABLE A — Metadata embeddings (Pass 1)",
        train_df, test_map, lt_test_map, all_users,
        profile_emb=emb_meta,
        item_ids=item_ids, C=C, item_index=item_index,
        popularity=popularity,
    )

    # ── RUN B: review profile + metadata candidates ───────────────────────────
    results_b = run_evaluation(
        "TABLE B — User-review enriched embeddings (Pass 2)",
        train_df, test_map, lt_test_map, all_users,
        profile_emb=emb_review,
        item_ids=item_ids, C=C, item_index=item_index,
        popularity=popularity,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "results_amazon.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "system", "Recall@10", "LT-Recall@10", "LT/Content Ratio"])
        for tbl, results in [("A_metadata", results_a), ("B_user_review", results_b)]:
            for sys in ["CF", "Content", "Essence"]:
                r = results[sys]
                w.writerow([tbl, sys,
                    f"{r['Recall@10']:.4f}", f"{r['LT-Recall@10']:.4f}",
                    f"{r['LT/Content Ratio']:.4f}" if not np.isnan(r["LT/Content Ratio"]) else "nan"])
    print(f"\nResults saved to {out_path}")

    # ── CHECKPOINT 4 ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("CHECKPOINT 4 — Final results")
    print("="*65)
    print_table("TABLE A — Metadata embeddings (Pass 1)", results_a)
    print_table("TABLE B — User-review enriched embeddings (Pass 2)", results_b)

    print("\n--- Pattern check: Essence > Content > CF on LT-Recall? ---")
    for tbl, results in [("A (Metadata)", results_a), ("B (User-review)", results_b)]:
        cf  = results["CF"]["LT-Recall@10"]
        con = results["Content"]["LT-Recall@10"]
        ess = results["Essence"]["LT-Recall@10"]
        holds = ess > con > cf
        print(f"  {tbl}: {'YES ✓' if holds else 'NO ✗'}  CF={cf:.4f}  Content={con:.4f}  Essence={ess:.4f}")

    print("\n--- Pass 2 vs Pass 1 delta ---")
    for sys in ["Content", "Essence"]:
        for metric in ["Recall@10", "LT-Recall@10"]:
            a = results_a[sys][metric]
            b = results_b[sys][metric]
            print(f"  {sys} {metric}: {a:.4f} → {b:.4f}  (Δ {b-a:+.4f})")

    print("\nCHECKPOINT 4 COMPLETE — do not commit. Waiting for your review.")
    print("="*65)


if __name__ == "__main__":
    main()
