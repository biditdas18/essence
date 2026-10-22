"""
Does the win/loss pattern belong to K-means clustering, or to the whole
no-cross-user-pooling architecture family?

WHY THIS EXISTS
---------------
Essence's standing against collaborative baselines flips sign between sparse and
dense cohorts. If that pattern is a property of *clustering*, it should not
appear in a system with no clustering at all. Recency-Weighted uses the same
embeddings and the same cosine retrieval, with no K-means step. Running it
against the same four collaborative baselines, under the same paired bootstrap
and the same BH-FDR family structure, tests whether the pattern survives the
removal of the component the paper is about.

This analysis was previously computed ad hoc against the merged per-user CSVs
and reported in the README without a script behind it -- the one reproducibility
gap among the repository's findings. This file closes it.

The comparison is deliberately symmetric with evaluation/paired_bootstrap.py:
same resample count, same tie-aware two-sided p-value (see that file for why
counting ties in one tail only inflates significance), same BH correction, same
pooling of the sparse-cohort datasets into one family with the dense one
corrected separately.

Usage
-----
    python evaluation/recency_weighted_family_test.py
    python evaluation/recency_weighted_family_test.py --n-boot 10000 --seed 42

Writes results/recency_weighted_family_test.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Adopted K per dataset (see README, K-Selection Methodology). Pooling follows
# the paper: the two sparse-co-occurrence cohorts share a correction family; the
# dense one is corrected separately, because it asks a different question.
DATASETS = {
    "lastfm": dict(K=10, family="sparse"),
    "amazon": dict(K=8, family="sparse"),
    "movielens": dict(K=10, family="dense"),
}
COLLABORATIVE = ["CF (ItemKNN)", "Popularity", "MIND", "ComiRec (SA)"]
METRICS = ["recall@10", "long_tail_recall@10"]
PROBE = "Recency-Weighted"


def paired_bootstrap(a, b, n_boot, rng):
    """Paired bootstrap over users. Returns (observed diff, CI, d, two-sided p).

    The p-value counts exact ties in BOTH tails. With most users scoring zero on
    long-tail metrics the resample distribution has heavy mass at zero, and
    counting it on one side only reports a one-sided-with-ties difference as
    maximally significant.
    """
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = diff.std(ddof=1)
    d = float(observed / sd) if sd > 0 else float("nan")
    p = min(1.0, 2 * min(float(np.mean(boot >= 0)), float(np.mean(boot <= 0))))
    return observed, float(lo), float(hi), d, p


def bh(pvals):
    """Benjamini-Hochberg. Returns q-values in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for ds, cfg in DATASETS.items():
        path = RESULTS_DIR / f"scratch_evaluation_results_K{cfg['K']}_{ds}_merged.csv"
        if not path.exists():
            print(f"  skip {ds}: {path.name} not found")
            continue
        pu = pd.read_csv(path)
        for metric in METRICS:
            if metric not in pu.columns:
                continue
            probe = pu[pu.system == PROBE].set_index("user_id")[metric].dropna()
            for baseline in COLLABORATIVE:
                base = pu[pu.system == baseline].set_index("user_id")[metric].dropna()
                common = probe.index.intersection(base.index)
                if len(common) == 0:
                    continue
                a = probe.loc[common].to_numpy(float)
                b = base.loc[common].to_numpy(float)
                rng = np.random.default_rng(args.seed)
                obs, lo, hi, d, p = paired_bootstrap(a, b, args.n_boot, rng)
                rows.append(dict(
                    dataset=ds, family=cfg["family"], K=cfg["K"], metric=metric,
                    baseline=baseline, n_users=len(common),
                    recency_weighted_mean=float(a.mean()), baseline_mean=float(b.mean()),
                    observed_diff=obs, ci_lo=lo, ci_hi=hi, cohens_d=d, p_two_sided=p,
                ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no comparisons computed")
        return

    # BH within each family, matching the paper's structure
    df["q_value_bh"] = np.nan
    for fam, sub in df.groupby("family"):
        df.loc[sub.index, "q_value_bh"] = bh(sub.p_two_sided.to_numpy())
    df["significant_after_fdr"] = df.q_value_bh < 0.05
    df["verdict"] = np.where(
        ~df.significant_after_fdr, "not significant",
        np.where(df.observed_diff > 0, "Recency-Weighted ahead", "baseline ahead"))

    out = RESULTS_DIR / "recency_weighted_family_test.csv"
    df.to_csv(out, index=False)

    print(f"\n{len(df)} comparisons -> {out.name}\n")
    for fam in ("sparse", "dense"):
        sub = df[df.family == fam]
        if sub.empty:
            continue
        print(f"  {fam}-co-occurrence family ({', '.join(sorted(sub.dataset.unique()))}):")
        for metric in METRICS:
            s = sub[sub.metric == metric]
            if s.empty:
                continue
            ahead = (s.verdict == "Recency-Weighted ahead").sum()
            behind = (s.verdict == "baseline ahead").sum()
            print(f"    {metric:22} Recency-Weighted ahead {ahead}/{len(s)}, "
                  f"baseline ahead {behind}/{len(s)}")
    print()


if __name__ == "__main__":
    main()
