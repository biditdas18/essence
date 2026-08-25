"""
evaluation/step6c_full_chain_lastfm_scratch.py
-------------------------------------------------
Tier-2 Step 6c: real (not estimated) wall-clock timing of the FULL
downstream analysis chain -- bootstrap, FDR, decile, silhouette
stratification, active-cluster-selection ablation -- re-run at the
validation-selected K for Last.fm-1K (K=10, from Step 6b).

SCRATCH ONLY: every output uses a _K10_SCRATCH suffix. Nothing here
overwrites any committed K=3 file. This script exists purely to produce
one real timing number; it is not a permanent addition to the pipeline.

Run:
    python evaluation/step6c_full_chain_lastfm_scratch.py
"""

import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import essence_recommend, essence_recommend_meanselect, content_recommend
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

RESULTS_DIR = BASE_DIR / "results"
K = 10
SEED = 42
M = 10
N_BOOT = 10_000

stage_times = {}


def timed(name, fn):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    stage_times[name] = dt
    print(f"\n[step6c] STAGE '{name}' took {dt:.1f}s")
    return result


# ── Stage 1: full 10-system per-user evaluation at K=10 ──────────────────

def stage_evaluate():
    out_path = RESULTS_DIR / "scratch_evaluation_results_K10_lastfm.csv"
    subprocess.run(
        [sys.executable, "evaluation/evaluate.py", "--K", str(K), "--users", "99",
         "--output", str(out_path)],
        cwd=str(BASE_DIR), check=True,
    )
    base_df = pd.read_csv(out_path)
    mind_comirec = pd.read_csv(RESULTS_DIR / "mind_comirec_results_lastfm.csv")
    merged = pd.concat([base_df, mind_comirec], ignore_index=True)
    merged_path = RESULTS_DIR / "scratch_evaluation_results_K10_lastfm_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[step6c] merged 10-system CSV: {len(merged)} rows -> {merged_path}")
    return merged_path


# ── Stage 2: paired bootstrap ─────────────────────────────────────────────

def stage_bootstrap(merged_path):
    out_path = RESULTS_DIR / "scratch_paired_bootstrap_lastfm_K10.csv"
    subprocess.run(
        [sys.executable, "evaluation/paired_bootstrap.py",
         "--input", str(merged_path), "--label", "Last.fm-1K (K=10)",
         "--output", str(out_path)],
        cwd=str(BASE_DIR), check=True,
    )
    return out_path


# ── Stage 3: FDR correction (Last.fm-only family, scratch) ───────────────

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
    out_path = RESULTS_DIR / "scratch_fdr_lastfm_K10.csv"
    df.to_csv(out_path, index=False)
    print(f"[step6c] {df['significant_after_fdr'].sum()}/{m} significant after FDR")
    return out_path


# ── Stage 4: decile analysis (K=10, Last.fm) ──────────────────────────────

def stage_decile():
    from models.recommenders import (
        build_itemknn_model, cf_itemknn_recommend, random_recommend,
        popularity_recommend, last_item_recommend, avg_last10_recommend,
        recency_weighted_recommend,
    )
    from collections import defaultdict
    import hashlib

    train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
    test_df = pd.read_pickle(BASE_DIR / "data" / "test_interactions.pkl")
    with open(BASE_DIR / "embeddings" / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)

    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))
    item_ids_all = sorted(emb.keys())
    popularity = train_df.groupby("track_id").size()
    pop_series = pd.Series(0, index=item_ids_all, dtype=int)
    pop_series.update(popularity)
    ranks = pop_series.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False) + 1
    item_decile = dict(zip(pop_series.index, deciles))
    itemknn = build_itemknn_model(train_df, item_col="track_id")

    ess_name = f"Essence (K={K})"
    systems = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)",
               "Last-Item", "Avg-Last-10", "Recency-Weighted", ess_name]
    per_system_decile_recalls = {s: defaultdict(list) for s in systems}

    def _stable_user_seed(uid):
        return int.from_bytes(hashlib.md5(str(uid).encode()).digest()[:4], "big")

    for uid in all_users:
        actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
        if not actual:
            continue
        recs = {
            "Random": random_recommend(uid, train_df, M, seed=_stable_user_seed(uid), item_embedding_map=emb),
            "Popularity": popularity_recommend(uid, train_df, M, item_embedding_map=emb),
            "CF (ItemKNN)": cf_itemknn_recommend(uid, train_df, itemknn, M, item_embedding_map=emb),
            "Content (Avg Emb)": content_recommend(uid, train_df, emb, M),
            "Last-Item": last_item_recommend(uid, train_df, emb, M),
            "Avg-Last-10": avg_last10_recommend(uid, train_df, emb, M),
            "Recency-Weighted": recency_weighted_recommend(uid, train_df, emb, M),
            ess_name: essence_recommend(uid, train_df, emb, K=K, M=M),
        }
        test_by_decile = defaultdict(set)
        for iid in actual:
            test_by_decile[item_decile.get(iid, 1)].add(iid)
        for sys_name, rec_list in recs.items():
            for d, items_in_decile in test_by_decile.items():
                r = recall_at_k(rec_list, list(items_in_decile), k=M)
                per_system_decile_recalls[sys_name][d].append(r)

    decile_rows = []
    for d in range(1, 11):
        for s in systems:
            vals = per_system_decile_recalls[s].get(d, [])
            mean_r = float(np.mean(vals)) if vals else float("nan")
            decile_rows.append({"decile": d, "system": s, "recall@10": mean_r, "n_opportunities": len(vals)})
    out_path = RESULTS_DIR / "scratch_decile_recall_lastfm_K10.csv"
    pd.DataFrame(decile_rows).to_csv(out_path, index=False)
    return out_path


# ── Stage 5: silhouette / clustering-mechanism stratification (K=10) ─────

def stage_silhouette(merged_path):
    train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
    with open(BASE_DIR / "embeddings" / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    sil = {}
    for uid, g in train_df.groupby("user_id"):
        seen = list(g.sort_values("timestamp")["track_id"])
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) < K + 1:
            continue
        X = np.array(vecs)
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sil[uid] = silhouette_score(X, km.labels_)
    print(f"[step6c] silhouette computed for {len(sil)} users at K={K}")

    df = pd.read_csv(merged_path)
    ess_name = f"Essence (K={K})"
    essence = df[df["system"] == ess_name].set_index("user_id")
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
    out_path = RESULTS_DIR / "scratch_multimodality_stratified_lastfm_K10.csv"
    sil_df.to_csv(out_path, index=False)
    return out_path


# ── Stage 6: active-cluster-selection ablation (K=10) ─────────────────────

def stage_ablation():
    train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
    test_df = pd.read_pickle(BASE_DIR / "data" / "test_interactions.pkl")
    with open(BASE_DIR / "embeddings" / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    all_users = sorted(set(train_df["user_id"].unique()) & set(test_df["user_id"].unique()))

    rows = []
    for uid in all_users:
        actual = test_df[test_df["user_id"] == uid]["track_id"].tolist()
        recs_recency = essence_recommend(uid, train_df, emb, K=K, M=M)
        recs_mean = essence_recommend_meanselect(uid, train_df, emb, K=K, M=M)
        rows.append({
            "user_id": uid,
            "recency_recall@10": recall_at_k(recs_recency, actual, k=M),
            "meanselect_recall@10": recall_at_k(recs_mean, actual, k=M),
        })
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scratch_ablation_lastfm_K10.csv"
    out.to_csv(out_path, index=False)
    print(f"[step6c] Recency-select mean Recall@10={out['recency_recall@10'].mean():.4f}  "
          f"Mean-select mean Recall@10={out['meanselect_recall@10'].mean():.4f}")
    return out_path


def main():
    t_start = time.time()
    print(f"{'='*70}\nSTEP 6c: full downstream chain, Last.fm-1K, K={K} (validation-selected)\n{'='*70}")

    merged_path = timed("1_evaluate_10systems", stage_evaluate)
    bootstrap_path = timed("2_paired_bootstrap", lambda: stage_bootstrap(merged_path))
    timed("3_fdr_correction", lambda: stage_fdr(bootstrap_path))
    timed("4_decile_analysis", stage_decile)
    timed("5_silhouette_stratification", lambda: stage_silhouette(merged_path))
    timed("6_active_cluster_ablation", stage_ablation)

    total = time.time() - t_start
    print(f"\n{'='*70}\nSTEP 6c TOTAL: {total:.1f}s ({total/60:.1f} min)\n{'='*70}")
    for name, dt in stage_times.items():
        print(f"  {name:<32} {dt:>8.1f}s  ({dt/total*100:>5.1f}%)")


if __name__ == "__main__":
    main()
