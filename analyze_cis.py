"""
analyze_cis.py — Bootstrap 95% CIs for Recall@10 and LT-Recall@10.
Reads:
  results/evaluation_results_v5.csv          (Last.fm-1K, per-user)
  experiments/amazon_books/results_amazon_peruser.csv  (Amazon Books, per-user)
Bootstrap: seed=42, n_resamples=10000.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

RNG = np.random.default_rng(42)
N_BOOT = 10_000

SYSTEMS = ["Random", "Popularity", "CF (ItemKNN)", "Content (Avg Emb)", "Essence (K=3)"]


def bootstrap_ci(values, n_boot=N_BOOT, rng=RNG):
    arr = np.array(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return arr.mean(), lo, hi


def load_peruser(path):
    """Returns dict: system -> {user_id -> {recall, lt or None}}"""
    data = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row["user_id"]
            sys = row["system"]
            r   = float(row["recall@10"])
            lt_raw = row["long_tail_recall@10"]
            lt  = float(lt_raw) if lt_raw.strip() != "" else None
            data[sys][uid] = {"recall": r, "lt": lt}
    return data


def report(dataset_label, data):
    print(f"\n{'='*90}")
    print(f"  {dataset_label}")
    print(f"{'='*90}")

    # header
    hdr = (f"  {'System':<22} {'R@10 mean':>10} {'95% CI':>22}  "
           f"{'n':>5} {'n_hit':>6}  |  "
           f"{'LT mean':>9} {'95% CI':>22}  {'n_elig':>6} {'n_LThit':>7}")
    print(hdr)
    print("  " + "-" * 110)

    for sys in SYSTEMS:
        umap = data.get(sys, {})
        all_users = sorted(umap.keys())
        n = len(all_users)

        recalls = [umap[u]["recall"] for u in all_users]
        n_hit   = sum(1 for v in recalls if v > 0)
        r_mean, r_lo, r_hi = bootstrap_ci(recalls)

        lt_vals = [umap[u]["lt"] for u in all_users if umap[u]["lt"] is not None]
        n_elig  = len(lt_vals)
        n_lt_hit = sum(1 for v in lt_vals if v > 0)
        if n_elig > 0:
            lt_mean, lt_lo, lt_hi = bootstrap_ci(lt_vals)
            lt_str  = f"{lt_mean:.4f}"
            ltci_str = f"[{lt_lo:.4f}, {lt_hi:.4f}]"
        else:
            lt_str = lt_mean = "  —  "
            ltci_str = "        —        "
            n_lt_hit = 0

        ci_str = f"[{r_lo:.4f}, {r_hi:.4f}]"
        print(f"  {sys:<22} {r_mean:>10.4f} {ci_str:>22}  "
              f"{n:>5} {n_hit:>6}  |  "
              f"{lt_str:>9} {ltci_str:>22}  {n_elig:>6} {n_lt_hit:>7}")

    print()


def main():
    base = Path(__file__).parent

    # ── Last.fm-1K ──────────────────────────────────────────────────────────
    lastfm_path = base / "results" / "evaluation_results_v5.csv"
    lastfm_data = load_peruser(lastfm_path)
    report("Last.fm-1K  (99 users · |I|=22,767 · chronological 80/20)", lastfm_data)

    # ── Amazon Books ────────────────────────────────────────────────────────
    amazon_path = base / "experiments" / "amazon_books" / "results_amazon_peruser.csv"
    amazon_data = load_peruser(amazon_path)
    report("Amazon Books  (2,000 users · |I|=61,727 · chronological 80/20)", amazon_data)


if __name__ == "__main__":
    main()
