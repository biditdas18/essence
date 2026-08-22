"""
evaluation/clustering_value_test.py
---------------------------------------
Step 4: does clustering help conditional on how multi-modal a user's taste
actually is? Isolates clustering's specific contribution by comparing
Essence (clustering + recency-based active-centroid selection) against
Recency-Weighted (same embeddings, same recency emphasis, NO clustering --
a single exponentially-weighted query vector) -- the cleanest baseline
pair for testing whether clustering itself adds value beyond recency
weighting alone.

Hypothesis: users whose history is genuinely multi-modal (high silhouette
score under their own K=3 fit) should show Essence matching or beating
Recency-Weighted; users whose history is more unimodal (low silhouette)
should show Essence losing more clearly, since clustering a unimodal cloud
into 3 pieces just fragments a single coherent taste into noisy sub-parts.

Silhouette scores are computed fresh per user (never cached anywhere) by
refitting the exact same K=3 KMeans fit (same seed=42) used by
essence_recommend, on each user's own train embeddings.

Per-user Recall@10 / LT-Recall@10 for Essence and Recency-Weighted are
reused from the existing canonical per-user result files (not
recomputed) -- only the silhouette stratification is new.

Saves:
  results/multimodality_stratified_lastfm.csv
  results/multimodality_stratified_amazon.csv
  results/multimodality_stratified_movielens.csv
  results/clustering_value_test.md

Run:
    python evaluation/clustering_value_test.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
K = 3
SEED = 42
N_BOOT = 10_000


def two_sided_p(f):
    return min(1.0, 2 * min(f, 1 - f))


def paired_bootstrap(a_vals, b_vals, rng, n_boot=N_BOOT):
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    n = len(a_vals)
    diff = a_vals - b_vals
    observed = diff.mean()
    diff_std = diff.std(ddof=1) if n > 1 else float("nan")
    d = observed / diff_std if diff_std > 0 else float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a_vals[idx].mean(axis=1) - b_vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac_gt0 = float(np.mean(boot > 0))
    return dict(a_mean=a_vals.mean(), b_mean=b_vals.mean(), n=n, observed_diff=observed,
               ci_lo=lo, ci_hi=hi, cohens_d=d, frac_gt0=frac_gt0, p_two_sided=two_sided_p(frac_gt0))


def compute_silhouettes_lastfm():
    train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
    with open(BASE_DIR / "embeddings" / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    sil = {}
    for uid, g in tqdm(train_df.groupby("user_id"), desc="Last.fm silhouettes"):
        seen = list(g.sort_values("timestamp")["track_id"])
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) < K + 1:  # silhouette needs > K samples
            continue
        X = np.array(vecs)
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sil[uid] = silhouette_score(X, km.labels_)
    return sil


def compute_silhouettes_amazon():
    train_df = pd.read_csv(BASE_DIR / "data" / "amazon_processed" / "train.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(BASE_DIR / "data" / "amazon_processed" / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    sil = {}
    for uid, g in tqdm(train_df.groupby("user_id"), desc="Amazon silhouettes"):
        seen = list(g.sort_values("timestamp")["item_id"])
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) < K + 1:
            continue
        X = np.array(vecs)
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sil[uid] = silhouette_score(X, km.labels_)
    return sil


def compute_silhouettes_movielens():
    train_df = pd.read_csv(BASE_DIR / "data" / "movielens_processed" / "train.csv")
    with open(BASE_DIR / "data" / "movielens_processed" / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    sil = {}
    for uid, g in tqdm(train_df.groupby("user_id"), desc="MovieLens silhouettes"):
        seen = list(g.sort_values("timestamp")["item_id"])
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) < K + 1:
            continue
        X = np.array(vecs)
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        sil[uid] = silhouette_score(X, km.labels_)
    return sil


def process_dataset(label, peruser_path, sil_fn, tag=""):
    print(f"\n=== {label} ===")
    sil = sil_fn()
    print(f"  Computed silhouette for {len(sil)} users")

    df = pd.read_csv(peruser_path)
    essence = df[df["system"] == "Essence (K=3)"].set_index("user_id")
    recency = df[df["system"] == "Recency-Weighted"].set_index("user_id")

    rows = []
    for uid, s in sil.items():
        if uid not in essence.index or uid not in recency.index:
            continue
        e_r = essence.loc[uid, "recall@10"]
        r_r = recency.loc[uid, "recall@10"]
        e_lt = pd.to_numeric(pd.Series([essence.loc[uid, "long_tail_recall@10"]]), errors="coerce").iloc[0]
        r_lt = pd.to_numeric(pd.Series([recency.loc[uid, "long_tail_recall@10"]]), errors="coerce").iloc[0]
        rows.append({"user_id": uid, "silhouette": s, "essence_recall": e_r, "recency_recall": r_r,
                    "essence_lt": e_lt, "recency_lt": r_lt})

    sil_df = pd.DataFrame(rows)
    sil_df["stratum"] = pd.qcut(sil_df["silhouette"], 3, labels=["low", "medium", "high"])
    out_path = RESULTS_DIR / f"multimodality_stratified_{tag}.csv"
    sil_df.to_csv(out_path, index=False)
    print(f"  Saved {len(sil_df)} users to {out_path}")

    rng = np.random.default_rng(SEED)
    results = []
    for stratum in ["low", "medium", "high"]:
        sub = sil_df[sil_df["stratum"] == stratum]
        sil_range = f"[{sub['silhouette'].min():.3f}, {sub['silhouette'].max():.3f}]"
        for metric, e_col, r_col in [("Recall@10", "essence_recall", "recency_recall"),
                                     ("LT-Recall@10", "essence_lt", "recency_lt")]:
            valid = sub.dropna(subset=[e_col, r_col])
            if len(valid) < 5:
                print(f"  {stratum:>6} {metric:<14}: n={len(valid)} too small, skipping")
                continue
            res = paired_bootstrap(valid[e_col].values, valid[r_col].values, rng)
            res.update({"dataset": label, "stratum": stratum, "metric": metric, "silhouette_range": sil_range})
            results.append(res)
            print(f"  {stratum:>6} {metric:<14}: n={res['n']:>5}  Essence={res['a_mean']:.4f}  "
                  f"Recency-W={res['b_mean']:.4f}  diff={res['observed_diff']:+.4f}  "
                  f"CI=[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  d={res['cohens_d']:+.3f}  p={res['p_two_sided']:.4f}")

    return results


def benjamini_hochberg(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_raw = ranked * m / np.arange(1, m + 1)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0, 1)
    q = np.empty(m)
    q[order] = q_mono
    return q, q <= alpha


def main():
    all_results = []
    all_results += process_dataset("Last.fm-1K", RESULTS_DIR / "evaluation_results_v6.csv",
                                    compute_silhouettes_lastfm, tag="lastfm")
    all_results += process_dataset("Amazon Books", BASE_DIR / "experiments" / "amazon_books" / "results_amazon_peruser.csv",
                                    compute_silhouettes_amazon, tag="amazon")
    all_results += process_dataset("MovieLens-25M", RESULTS_DIR / "results_movielens_peruser_full.csv",
                                    compute_silhouettes_movielens, tag="movielens")

    res_df = pd.DataFrame(all_results)
    q, reject = benjamini_hochberg(res_df["p_two_sided"].values)
    res_df["q_value_bh"] = q
    res_df["significant_after_fdr"] = reject
    res_df = res_df.sort_values("p_two_sided")
    res_df.to_csv(RESULTS_DIR / "clustering_value_test_fdr.csv", index=False)

    print(f"\n{'='*100}")
    print(f"FDR-corrected results ({len(res_df)} comparisons, one family)")
    print(f"{'='*100}")
    print(res_df[["dataset", "stratum", "metric", "observed_diff", "cohens_d", "p_two_sided", "q_value_bh", "significant_after_fdr"]].to_string(index=False))

    # Markdown summary
    lines = ["# Clustering Value Test — does clustering help conditional on taste multi-modality?\n",
            "Essence (clustering) vs. Recency-Weighted (no clustering, same embeddings, same recency emphasis),",
            "stratified by per-user silhouette score under each user's own K=3 fit (tertiles, computed per dataset).",
            f"FDR family size: {len(res_df)} (3 datasets x up to 3 strata x 2 metrics).\n"]

    for label in ["Last.fm-1K", "Amazon Books", "MovieLens-25M"]:
        sub = res_df[res_df["dataset"] == label]
        lines.append(f"\n## {label}\n")
        lines.append("| Stratum | Metric | n | Essence | Recency-W | Diff | Cohen's d | p | Sig. (FDR) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.sort_values(["metric", "stratum"]).iterrows():
            lines.append(f"| {r['stratum']} | {r['metric']} | {r['n']} | {r['a_mean']:.4f} | {r['b_mean']:.4f} | "
                        f"{r['observed_diff']:+.4f} | {r['cohens_d']:+.3f} | {r['p_two_sided']:.4f} | "
                        f"{'Yes' if r['significant_after_fdr'] else 'No'} |")

    with open(RESULTS_DIR / "clustering_value_test.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary to {RESULTS_DIR / 'clustering_value_test.md'}")


if __name__ == "__main__":
    main()
