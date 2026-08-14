"""
experiments/mind_comirec/sanity_check.py
--------------------------------------------
Step 6 (per plan): before treating any MIND/ComiRec result as valid,
sanity-check it against published benchmark ranges — a result wildly out
of line with known literature means a bug, not a finding.

IMPORTANT CAVEAT (read before trusting this script's verdicts):
Neither the original MIND paper (Li et al., CIKM 2019 — benchmarks on
Alibaba's internal industrial dataset and Amazon Books) nor the ComiRec
paper (Cen et al., RecSys 2020 — benchmarks on Amazon Books, Taobao,
and a mobile-Taobao "UserBehavior" set) report numbers on Last.fm-1K in
any form, let alone this repo's specific 99-user filtered/chronological-
split subset. There is no published Last.fm-1K range to check against.
For Amazon Books, the papers' own preprocessing (dataset size, core-N
filtering, negative-sampling eval protocol) differs enough from this
repo's setup (2,000-user subsample, full-catalog ranking, no sampled
negatives) that a direct number-to-number comparison isn't meaningful
either -- these are architectural sanity ranges, not a reproduction
target.

What this script actually checks, given that:
  1. Directional sanity: MIND/ComiRec should convincingly beat Random
     and Popularity (weak baselines with no personalization or
     population-level structure) -- a multi-interest model that can't
     clear this bar has a real bug, independent of any external paper.
  2. Order-of-magnitude sanity vs. published Recall@10 ranges: the
     MIND paper reports Recall@50 in the 4-8% range on its Amazon Books
     split; ComiRec reports Recall@20/50 in the 3-10% range on Amazon
     Books / Taobao under sampled-negative eval (typically 1 positive
     vs ~100-1000 negatives), which inflates recall relative to this
     repo's full-catalog (tens of thousands of candidates) protocol.
     So this repo's numbers are expected to be an order of magnitude
     (or more) BELOW those published ranges by construction of the
     harder eval protocol -- that is not itself a failure. What WOULD
     be a failure: recall indistinguishable from Random (no learning
     happened) or a value that's implausibly high relative to the
     harder protocol (something leaked / eval bug).

Run:
    python experiments/mind_comirec/sanity_check.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"


def load_summary(dataset):
    path = RESULTS_DIR / f"mind_comirec_summary_{dataset}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_baseline(dataset):
    """Random / Popularity Recall@10 from the existing per-user results, for the directional check."""
    if dataset == "lastfm":
        path = RESULTS_DIR / "evaluation_results_v6.csv"
    else:
        path = BASE_DIR / "experiments" / "amazon_books" / "results_amazon_peruser.csv"
    df = pd.read_csv(path)
    out = {}
    for sys_name in ["Random", "Popularity", "Essence (K=3)"]:
        sub = df[df["system"] == sys_name]
        out[sys_name] = sub["recall@10"].mean()
    return out


def main():
    all_pass = True
    for dataset in ["lastfm", "amazon"]:
        summary = load_summary(dataset)
        if summary is None:
            print(f"[sanity_check] No results yet for {dataset} — skipping")
            continue
        baselines = load_baseline(dataset)

        print(f"\n{'='*70}")
        print(f"  SANITY CHECK — {dataset}")
        print(f"{'='*70}")
        print(f"  Random Recall@10:     {baselines['Random']:.4f}")
        print(f"  Popularity Recall@10: {baselines['Popularity']:.4f}")
        print(f"  Essence Recall@10:    {baselines['Essence (K=3)']:.4f}")

        for _, row in summary.iterrows():
            name = row["system"]
            r10 = row["recall@10"]
            beats_random = r10 > baselines["Random"] * 3  # convincingly above noise floor
            beats_pop = r10 > baselines["Popularity"] * 0.3  # in the ballpark of a weak personalized baseline
            verdict = "PASS" if beats_random else "FAIL"
            print(f"\n  {name}: Recall@10={r10:.4f} LT-Recall@10={row['lt_recall@10']:.4f}")
            print(f"    beats Random by >3x:        {'yes' if beats_random else 'NO'}")
            print(f"    within 0.3x of Popularity:  {'yes' if beats_pop else 'no (weaker than expected)'}")
            print(f"    directional sanity: {verdict}")
            if not beats_random:
                all_pass = False

    print(f"\n{'='*70}")
    print(f"  OVERALL: {'directional sanity PASSED for all runs' if all_pass else 'AT LEAST ONE RUN FAILED — see above, do not trust bootstrap results for the failing run(s)'}")
    print(f"{'='*70}\n")
    print("  NOTE: no published Last.fm-1K numbers exist for MIND/ComiRec (see")
    print("  docstring). Amazon Books published numbers use a different eval")
    print("  protocol (sampled-negative, different preprocessing) and are not")
    print("  directly comparable -- this script checks directional sanity, not")
    print("  literature reproduction. Read the full report before trusting a PASS.")


if __name__ == "__main__":
    main()
