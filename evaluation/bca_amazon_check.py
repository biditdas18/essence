"""
evaluation/bca_amazon_check.py
---------------------------------
Step 3 robustness check: re-run the Essence-vs-MIND and Essence-vs-ComiRec
Amazon LT-Recall@10 comparisons (the paper's central claim) using BCa
(bias-corrected and accelerated) bootstrap instead of the plain percentile
method paired_bootstrap.py uses, via scipy.stats.bootstrap(method='BCa').
Reports whether the significance conclusion changes.

Run:
    python evaluation/bca_amazon_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "results" / "results_amazon_peruser_mind_comirec.csv"
OUTPUT = BASE_DIR / "results" / "bca_amazon_check.csv"

N_RESAMPLES = 10_000
SEED = 42


def load_peruser(path):
    df = pd.read_csv(path)
    out = {}
    for sys_name, sub in df.groupby("system"):
        out[sys_name] = {
            row["user_id"]: {
                "recall": row["recall@10"],
                "lt": None if pd.isna(row["long_tail_recall@10"]) or row["long_tail_recall@10"] == ""
                else float(row["long_tail_recall@10"]),
            }
            for _, row in sub.iterrows()
        }
    return out


def paired_diff_stat(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def main():
    data = load_peruser(INPUT)
    essence = data["Essence (K=3)"]

    rows = []
    for baseline_name in ["MIND", "ComiRec (SA)"]:
        baseline = data[baseline_name]
        common = sorted(
            u for u in (set(essence) & set(baseline))
            if essence[u]["lt"] is not None and baseline[u]["lt"] is not None
        )
        e_vals = np.array([essence[u]["lt"] for u in common])
        b_vals = np.array([baseline[u]["lt"] for u in common])

        observed_diff = e_vals.mean() - b_vals.mean()

        # Percentile bootstrap (matches paired_bootstrap.py's method)
        rng_pct = np.random.default_rng(SEED)
        idx = rng_pct.integers(0, len(e_vals), size=(N_RESAMPLES, len(e_vals)))
        pct_diffs = e_vals[idx].mean(axis=1) - b_vals[idx].mean(axis=1)
        pct_lo, pct_hi = np.percentile(pct_diffs, [2.5, 97.5])
        pct_sig = not (pct_lo <= 0 <= pct_hi)

        # BCa bootstrap via scipy
        res = bootstrap(
            (e_vals, b_vals), paired_diff_stat, n_resamples=N_RESAMPLES,
            paired=True, vectorized=True, method="BCa",
            confidence_level=0.95, rng=SEED,
        )
        bca_lo, bca_hi = res.confidence_interval.low, res.confidence_interval.high
        bca_sig = not (bca_lo <= 0 <= bca_hi)

        print(f"\n--- Essence vs {baseline_name} (Amazon, LT-Recall@10, n={len(common)}) ---")
        print(f"  Observed diff: {observed_diff:+.4f}")
        print(f"  Percentile 95% CI: [{pct_lo:+.4f}, {pct_hi:+.4f}]  significant={pct_sig}")
        print(f"  BCa       95% CI: [{bca_lo:+.4f}, {bca_hi:+.4f}]  significant={bca_sig}")
        print(f"  CONCLUSION CHANGES: {pct_sig != bca_sig}")

        rows.append({
            "baseline": baseline_name, "metric": "LT-Recall@10", "dataset": "Amazon Books",
            "observed_diff": observed_diff, "n": len(common),
            "percentile_ci_lo": pct_lo, "percentile_ci_hi": pct_hi, "percentile_significant": pct_sig,
            "bca_ci_lo": bca_lo, "bca_ci_hi": bca_hi, "bca_significant": bca_sig,
            "conclusion_changes": pct_sig != bca_sig,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")

    if out_df["conclusion_changes"].any():
        print("\n*** WARNING: at least one conclusion CHANGES under BCa. This needs to be flagged, not footnoted. ***")
    else:
        print("\nNo conclusions change under BCa -- one-line robustness footnote is warranted.")


if __name__ == "__main__":
    main()
