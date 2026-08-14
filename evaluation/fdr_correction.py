"""
evaluation/fdr_correction.py
-------------------------------
Benjamini-Hochberg FDR correction across every Essence-vs-baseline paired
bootstrap comparison computed so far (both datasets, both metrics).

Source of p-values
-------------------
paired_bootstrap.py reports "frac_resamples_diff_gt_0" = P(diff > 0), a
one-sided bootstrap quantity for the alternative "Essence > baseline".
This script converts it to the standard two-sided empirical bootstrap
p-value:
    p = 2 * min(P(diff > 0), P(diff < 0)), capped at 1.0
and applies BH correction to that two-sided family, since the underlying
question ("is there a difference at all, in either direction") is what a
multiple-comparisons correction should protect against.

Family used
------------
The 36 "main" comparisons in results/paired_bootstrap_{lastfm,amazon}_
mind_comirec.csv (9 baselines x 2 datasets x 2 metrics) -- this is the
complete comparison set that would appear in a results table. The
random-init ablation comparisons (results/paired_bootstrap_*_randominit.csv)
are a separate diagnostic/robustness check on a subset of these baselines,
not additional claims requiring their own slot in this correction family,
so they are excluded here (reported separately in their own files).

Run:
    python evaluation/fdr_correction.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

ALPHA = 0.05

SOURCES = [
    ("Last.fm-1K", RESULTS_DIR / "paired_bootstrap_lastfm_mind_comirec.csv"),
    ("Amazon Books", RESULTS_DIR / "paired_bootstrap_amazon_mind_comirec.csv"),
]


def two_sided_p(frac_gt_0: float) -> float:
    return min(1.0, 2 * min(frac_gt_0, 1 - frac_gt_0))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = ALPHA):
    """Returns (q_values, reject_mask) for the BH procedure."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    # BH critical values and raw q-value candidates
    q_raw = ranked * m / (np.arange(1, m + 1))
    # Enforce monotonicity (q-values must be non-decreasing from the largest p down)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_monotone = np.clip(q_monotone, 0, 1)

    q_values = np.empty(m)
    q_values[order] = q_monotone

    reject = q_values <= alpha
    return q_values, reject


def main():
    rows = []
    for label, path in SOURCES:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            p_two_sided = two_sided_p(row["frac_resamples_diff_gt_0"])
            rows.append({
                "dataset": label,
                "baseline": row["baseline_system"],
                "metric": row["metric"],
                "observed_diff": row["observed_diff"],
                "cohens_d_paired": row.get("cohens_d_paired", float("nan")),
                "frac_resamples_diff_gt_0": row["frac_resamples_diff_gt_0"],
                "p_value_raw_two_sided": p_two_sided,
            })

    all_df = pd.DataFrame(rows)
    q_values, reject = benjamini_hochberg(all_df["p_value_raw_two_sided"].values, ALPHA)
    all_df["q_value_bh"] = q_values
    all_df["significant_raw_p<0.05"] = all_df["p_value_raw_two_sided"] < 0.05
    all_df["significant_after_fdr"] = reject

    all_df = all_df.sort_values("p_value_raw_two_sided").reset_index(drop=True)

    out_path = RESULTS_DIR / "fdr_corrected_significance.csv"
    all_df.to_csv(out_path, index=False)

    n_sig_raw = all_df["significant_raw_p<0.05"].sum()
    n_sig_fdr = all_df["significant_after_fdr"].sum()
    n_flipped = ((all_df["significant_raw_p<0.05"]) & (~all_df["significant_after_fdr"])).sum()

    print(f"Family size (m): {len(all_df)}")
    print(f"Significant at raw p<0.05: {n_sig_raw}")
    print(f"Significant after BH-FDR (q<0.05): {n_sig_fdr}")
    print(f"Comparisons that LOSE significance after correction: {n_flipped}")
    print()
    print(all_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
