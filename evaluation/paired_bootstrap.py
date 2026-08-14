"""
evaluation/paired_bootstrap.py
-------------------------------
Paired bootstrap significance test: Essence vs every other baseline.

Replaces the CI-overlap comparison (analyze_cis.py computes independent
per-system 95% CIs and leaves the reader to eyeball whether they overlap)
with a direct test on the paired per-user difference, which is the
statistically correct way to ask "is Essence better than baseline X".

For each (Essence, baseline) pair and each metric (Recall@10, LT-Recall@10):
  - restrict to users present for both systems, paired by user_id
  - for LT-Recall@10, further restrict to users with a non-null LT value
    (a user has a null LT value iff they have zero long-tail items in
    their test set — this is a property of the user/test-set, not of
    the system, so it is shared across all systems for that user)
  - resample users with replacement 10,000 times (seed=42)
  - each resample: mean(essence_metric) - mean(baseline_metric) over the
    resampled users
  - report the observed diff, the 95% CI on the diff, and the fraction
    of resamples with diff > 0

Run:
    python evaluation/paired_bootstrap.py \
        --input results/evaluation_results_v6.csv \
        --label "Last.fm-1K" \
        --output results/paired_bootstrap_lastfm.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 42
ESSENCE_PREFIX = "Essence"


def load_peruser(path):
    """Returns dict: system -> {user_id -> {recall, lt or None}}"""
    data = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row["user_id"]
            sys_name = row["system"]
            r = float(row["recall@10"])
            lt_raw = row["long_tail_recall@10"]
            lt = float(lt_raw) if lt_raw.strip() != "" else None
            data[sys_name][uid] = {"recall": r, "lt": lt}
    return data


def paired_bootstrap_diff(essence_vals, baseline_vals, rng, n_boot=N_BOOT):
    """
    essence_vals, baseline_vals: parallel arrays (same user order).
    Returns (observed_diff, ci_lo, ci_hi, frac_gt_0).
    """
    essence_vals = np.asarray(essence_vals, dtype=float)
    baseline_vals = np.asarray(baseline_vals, dtype=float)
    n = len(essence_vals)

    observed_diff = essence_vals.mean() - baseline_vals.mean()

    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = essence_vals[idx].mean(axis=1) - baseline_vals[idx].mean(axis=1)

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    frac_gt_0 = float(np.mean(diffs > 0))
    return observed_diff, ci_lo, ci_hi, frac_gt_0


def run(input_csv: Path, label: str, output_csv: Path = None):
    data = load_peruser(input_csv)
    essence_systems = [s for s in data if s.startswith(ESSENCE_PREFIX)]
    if not essence_systems:
        raise ValueError(f"No system starting with '{ESSENCE_PREFIX}' found in {input_csv}")
    essence_name = essence_systems[0]
    baselines = [s for s in data if s != essence_name]

    rng = np.random.default_rng(SEED)
    rows_out = []

    print(f"\n{'='*100}")
    print(f"  Paired bootstrap: {essence_name} vs baselines — {label}")
    print(f"  ({N_BOOT:,} resamples, seed={SEED})")
    print(f"{'='*100}")

    for metric_key, metric_label in [("recall", "Recall@10"), ("lt", "LT-Recall@10")]:
        print(f"\n  --- {metric_label} ---")
        header = (f"  {'Baseline':<20} {'Essence mean':>13} {'Baseline mean':>14} "
                  f"{'Diff':>9} {'95% CI on diff':>22} {'P(diff>0)':>10} {'n':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for baseline_name in baselines:
            essence_users = data[essence_name]
            baseline_users = data[baseline_name]
            common_users = sorted(set(essence_users) & set(baseline_users))

            if metric_key == "lt":
                common_users = [
                    u for u in common_users
                    if essence_users[u]["lt"] is not None
                    and baseline_users[u]["lt"] is not None
                ]

            if not common_users:
                continue

            essence_vals = [essence_users[u][metric_key] for u in common_users]
            baseline_vals = [baseline_users[u][metric_key] for u in common_users]

            observed_diff, ci_lo, ci_hi, frac_gt_0 = paired_bootstrap_diff(
                essence_vals, baseline_vals, rng
            )
            e_mean = float(np.mean(essence_vals))
            b_mean = float(np.mean(baseline_vals))

            ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            print(f"  {baseline_name:<20} {e_mean:>13.4f} {b_mean:>14.4f} "
                  f"{observed_diff:>+9.4f} {ci_str:>22} {frac_gt_0:>10.3f} {len(common_users):>6}")

            rows_out.append({
                "dataset": label,
                "metric": metric_label,
                "essence_system": essence_name,
                "baseline_system": baseline_name,
                "essence_mean": f"{e_mean:.6f}",
                "baseline_mean": f"{b_mean:.6f}",
                "observed_diff": f"{observed_diff:.6f}",
                "ci_lo": f"{ci_lo:.6f}",
                "ci_hi": f"{ci_hi:.6f}",
                "frac_resamples_diff_gt_0": f"{frac_gt_0:.4f}",
                "n_paired_users": len(common_users),
            })

    print()

    if output_csv is not None:
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            w.writeheader()
            w.writerows(rows_out)
        print(f"[paired_bootstrap] Saved to {output_csv}\n")

    return rows_out


def parse_args():
    parser = argparse.ArgumentParser(description="Paired bootstrap: Essence vs each baseline")
    parser.add_argument("--input", type=str, required=True,
                        help="Per-user results CSV (user_id,system,recall@10,long_tail_recall@10)")
    parser.add_argument("--label", type=str, required=True,
                        help="Dataset label for the report header, e.g. 'Last.fm-1K'")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.output) if args.output else None
    run(Path(args.input), args.label, out)
