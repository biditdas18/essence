"""
evaluation/step6c_full_chain_movielens_scratch.py
------------------------------------------------------
Tier-2 Step 6c: real (not estimated) wall-clock timing of the FULL
downstream analysis chain -- bootstrap, FDR, decile, silhouette
stratification, active-cluster-selection ablation -- re-run at the
validation-selected K for MovieLens-25M (K=15, from Step 6b).

Vectorized throughout, mirroring
evaluation/step6c_full_chain_amazon_scratch.py's structure exactly.
Decile stage covers 5 systems (Essence, CF-ItemKNN, Last-Item,
Avg-Last-10, Recency-Weighted) -- a superset of MovieLens's existing
decile_analysis_movielens.py (Essence vs CF) and
decile_recency_vs_cf_movielens.py (recency baselines vs CF), giving full
cold-start coverage in one pass.

SCRATCH ONLY: every output uses a _K15_SCRATCH suffix. Nothing here
overwrites any committed K=3 file.

Run:
    python evaluation/step6c_full_chain_movielens_scratch.py
"""

import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
RESULTS_DIR = BASE_DIR / "results"

K = 15
SEED = 42
M = 10
N_DECILES = 10
RECENCY_N = 10
RECENCY_DECAY = 0.9
ESS_NAME = f"Essence (K={K})"

stage_times = {}


def timed(name, fn):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    stage_times[name] = dt
    print(f"\n[step6c-movielens] STAGE '{name}' took {dt:.1f}s")
    return result


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


def lt_recall_at_k(recs, test_items, lt_set):
    lt_test = {i for i in test_items if i in lt_set}
    if not lt_test:
        return None
    return sum(1 for r in recs if r in lt_test) / len(lt_test)


def _load_common():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)
    item_ids, C, item_index = build_candidate_matrix(emb_meta)
    return train_df, test_df, emb_meta, item_ids, C, item_index


# ── Stage 1: full 8-system per-user evaluation at K=15 (+ merge MIND/ComiRec) ──

def stage_evaluate():
    from models.recommenders import build_itemknn_model, cf_itemknn_recommend
    import hashlib

    train_df, test_df, emb_meta, item_ids, C, item_index = _load_common()
    n_items = len(item_ids)
    all_users = sorted(train_df["user_id"].unique())
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])

    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)
    itemknn = build_itemknn_model(train_df, item_col="item_id")

    def _stable_user_seed(uid):
        return int.from_bytes(hashlib.md5(str(uid).encode()).digest()[:4], "big")

    systems = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)",
               "Last-Item", "Avg-Last-10", "Recency-Weighted", ESS_NAME]
    rows = []

    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())

        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True

        unseen_pool = [i for i in item_ids if i not in seen_set]
        rng_u = np.random.default_rng(_stable_user_seed(uid))
        recs_random = rng_u.choice(unseen_pool, size=min(M, len(unseen_pool)), replace=False).tolist()
        recs_pop = [iid for iid in popularity.index if iid not in seen_set][:M]
        recs_knn = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if seen_vecs:
            recs_content = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs_last_item = top_k_unseen(normalized(seen_vecs[-1]), seen_mask, C, item_ids, M)
            recent_vecs = seen_vecs[-RECENCY_N:]
            recs_avg_last10 = top_k_unseen(normalized(np.mean(recent_vecs, axis=0)), seen_mask, C, item_ids, M)
            recs_recency_weighted = top_k_unseen(normalized(recency_weighted_query(recent_vecs)), seen_mask, C, item_ids, M)
            if len(seen_vecs) >= K:
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
                km.fit(np.array(seen_vecs))
                recent_mean = np.mean(recent_vecs, axis=0)
                dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
                centroid = km.cluster_centers_[np.argmin(dists)].astype(np.float32)
                recs_essence = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs_essence = recs_content
        else:
            recs_content = recs_last_item = recs_avg_last10 = recs_recency_weighted = recs_essence = recs_pop

        for name, recs in zip(systems, [recs_random, recs_pop, recs_knn, recs_content,
                                        recs_last_item, recs_avg_last10, recs_recency_weighted, recs_essence]):
            rows.append({
                "user_id": uid, "system": name,
                "recall@10": recall_at_k(recs, test_items),
                "long_tail_recall@10": lt_recall_at_k(recs, test_items, lt_set),
            })

    base_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scratch_evaluation_results_K15_movielens.csv"
    base_df.to_csv(out_path, index=False)

    mind_comirec = pd.read_csv(RESULTS_DIR / "mind_comirec_results_movielens.csv")
    merged = pd.concat([base_df, mind_comirec], ignore_index=True)
    merged_path = RESULTS_DIR / "scratch_evaluation_results_K15_movielens_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[step6c-movielens] merged 10-system CSV: {len(merged)} rows -> {merged_path}")
    return merged_path


# ── Stage 2/3: bootstrap + FDR ─────────────────────────────────────────────

def stage_bootstrap(merged_path):
    out_path = RESULTS_DIR / "scratch_paired_bootstrap_movielens_K15.csv"
    subprocess.run(
        [sys.executable, "evaluation/paired_bootstrap.py",
         "--input", str(merged_path), "--label", "MovieLens-25M (K=15)",
         "--output", str(out_path)],
        cwd=str(BASE_DIR), check=True,
    )
    return out_path


def stage_fdr(bootstrap_path):
    df = pd.read_csv(bootstrap_path)

    def two_sided_p(f):
        return min(1.0, 2 * min(f, 1 - f))

    df["p_two_sided"] = df["frac_resamples_diff_gt_0"].apply(two_sided_p)
    p = df["p_two_sided"].values
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_raw = ranked * m / np.arange(1, m + 1)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0, 1)
    q = np.empty(m)
    q[order] = q_mono
    df["q_value_bh"] = q
    df["significant_after_fdr"] = q <= 0.05
    out_path = RESULTS_DIR / "scratch_fdr_movielens_K15.csv"
    df.to_csv(out_path, index=False)
    print(f"[step6c-movielens] {df['significant_after_fdr'].sum()}/{m} significant after FDR")
    return out_path


# ── Stage 4: decile analysis (Essence, CF, 3 recency baselines, K=15) ─────

def stage_decile():
    from models.recommenders import build_itemknn_model, cf_itemknn_recommend

    train_df, test_df, emb_meta, item_ids, C, item_index = _load_common()
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

    systems = [ESS_NAME, "CF (ItemKNN)", "Last-Item", "Avg-Last-10", "Recency-Weighted"]
    per_system_decile_recalls = {s: defaultdict(list) for s in systems}

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
        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        recs = {"CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M)}
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
                recs[ESS_NAME] = top_k_unseen(normalized(centroid), seen_mask, C, item_ids, M)
            else:
                recs[ESS_NAME] = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            for s in ["Last-Item", "Avg-Last-10", "Recency-Weighted", ESS_NAME]:
                recs[s] = []

        test_by_decile = defaultdict(set)
        for iid in test_items:
            test_by_decile[item_decile.get(iid, 1)].add(iid)
        for sys_name, rec_list in recs.items():
            for d, items_in_decile in test_by_decile.items():
                r = recall_at_k(rec_list, items_in_decile)
                per_system_decile_recalls[sys_name][d].append(r)

    decile_rows = []
    for d in range(1, N_DECILES + 1):
        for s in systems:
            vals = per_system_decile_recalls[s].get(d, [])
            mean_r = float(np.mean(vals)) if vals else float("nan")
            decile_rows.append({"decile": d, "system": s, "recall@10": mean_r, "n_opportunities": len(vals)})
    out_path = RESULTS_DIR / "scratch_decile_recall_movielens_K15.csv"
    pd.DataFrame(decile_rows).to_csv(out_path, index=False)
    return out_path


# ── Stage 5: silhouette / clustering-mechanism stratification (K=15) ─────

def stage_silhouette(merged_path):
    train_df, test_df, emb_meta, item_ids, C, item_index = _load_common()
    sil = {}
    for uid, g in train_df.groupby("user_id"):
        seen = list(g.sort_values("timestamp")["item_id"])
        vecs = [emb_meta[i] for i in seen if i in emb_meta]
        if len(vecs) < K + 1:
            continue
        X = np.array(vecs)
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sil[uid] = silhouette_score(X, km.labels_)
    print(f"[step6c-movielens] silhouette computed for {len(sil)} users at K={K}")

    df = pd.read_csv(merged_path)
    essence = df[df["system"] == ESS_NAME].set_index("user_id")
    recency = df[df["system"] == "Recency-Weighted"].set_index("user_id")
    rows = []
    for uid, s in sil.items():
        if uid not in essence.index or uid not in recency.index:
            continue
        rows.append({
            "user_id": uid, "silhouette": s,
            "essence_recall": essence.loc[uid, "recall@10"],
            "recency_recall": recency.loc[uid, "recall@10"],
        })
    sil_df = pd.DataFrame(rows)
    if len(sil_df) >= 3:
        sil_df["stratum"] = pd.qcut(sil_df["silhouette"], 3, labels=["low", "medium", "high"], duplicates="drop")
    out_path = RESULTS_DIR / "scratch_multimodality_stratified_movielens_K15.csv"
    sil_df.to_csv(out_path, index=False)
    return out_path


# ── Stage 6: active-cluster-selection ablation (K=15) ─────────────────────

def stage_ablation():
    train_df, test_df, emb_meta, item_ids, C, item_index = _load_common()
    n_items = len(item_ids)
    test_map = defaultdict(set)
    for _, row in test_df.iterrows():
        test_map[row["user_id"]].add(row["item_id"])
    all_users = sorted(train_df["user_id"].unique())

    rows = []
    for uid in all_users:
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        test_items = test_map.get(uid, set())
        seen_mask = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
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
            recs_recency = recs_mean = top_k_unseen(normalized(np.mean(seen_vecs, axis=0)), seen_mask, C, item_ids, M)
        else:
            recs_recency = recs_mean = []
        rows.append({
            "user_id": uid,
            "recency_recall@10": recall_at_k(recs_recency, test_items),
            "meanselect_recall@10": recall_at_k(recs_mean, test_items),
        })
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scratch_ablation_movielens_K15.csv"
    out.to_csv(out_path, index=False)
    print(f"[step6c-movielens] Recency-select mean Recall@10={out['recency_recall@10'].mean():.4f}  "
          f"Mean-select mean Recall@10={out['meanselect_recall@10'].mean():.4f}")
    return out_path


def main():
    t_start = time.time()
    print(f"{'='*70}\nSTEP 6c: full downstream chain, MovieLens-25M, K={K} (validation-selected)\n{'='*70}")

    merged_path = timed("1_evaluate_8systems", stage_evaluate)
    bootstrap_path = timed("2_paired_bootstrap", lambda: stage_bootstrap(merged_path))
    timed("3_fdr_correction", lambda: stage_fdr(bootstrap_path))
    timed("4_decile_analysis", stage_decile)
    timed("5_silhouette_stratification", lambda: stage_silhouette(merged_path))
    timed("6_active_cluster_ablation", stage_ablation)

    total = time.time() - t_start
    print(f"\n{'='*70}\nSTEP 6c (MovieLens) TOTAL: {total:.1f}s ({total/60:.1f} min)\n{'='*70}")
    for name, dt in stage_times.items():
        print(f"  {name:<32} {dt:>8.1f}s  ({dt/total*100:>5.1f}%)")


if __name__ == "__main__":
    main()
