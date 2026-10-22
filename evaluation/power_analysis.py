"""
Minimum detectable effect (MDE) for the paired comparisons behind this
paper's null results.

WHY THIS EXISTS
---------------
The central claim of this work is a null: K-means interest clustering shows
no measurable benefit over recency-weighted retrieval. A null is only
informative if the test could have detected an effect had one existed.
Without that, "no significant difference" is indistinguishable from "not
enough data to tell", and a reader is right to discount the claim. This
script bounds the null by reporting, for every comparison, the smallest
true effect the test would have detected at 80% power.

METHOD
------
For a paired design the quantity of interest is the per-user difference
d_i = a_i - b_i. Under bootstrap resampling of users, the sampling
distribution of mean(d) has standard error SE_boot = std(bootstrap means).
Shifting every d_i by a constant delta shifts that distribution by delta
without changing its shape, so a two-sided test at alpha rejects
approximately when |delta| > z_{1-alpha/2} * SE_boot, and reaches power
(1-beta) when

    MDE = (z_{1-alpha/2} + z_{1-beta}) * SE_boot

With alpha = 0.05 and power = 0.80 the constant is 1.959964 + 0.841621
= 2.801585.

SE_boot is used rather than the normal-theory sd/sqrt(n) deliberately: the
long-tail metrics here are extremely tie-heavy (most users score exactly
zero), so the paired differences are far from normal and the bootstrap
sampling distribution is the honest description of their variability.

DEGENERATE CASES
----------------
If every per-user difference is exactly zero, SE_boot = 0 and the MDE
formula returns 0, which would absurdly imply infinite sensitivity. Those
comparisons are structurally uninformative -- neither system can register
a hit -- and are flagged rather than assigned an MDE. The same flag catches
strata where the metric has no non-zero observations at all.
"""

import numpy as np

Z_ALPHA_TWO_SIDED_05 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
MDE_CONST_80 = Z_ALPHA_TWO_SIDED_05 + Z_POWER_80

N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 42


def paired_mde(a_vals, b_vals, n_boot=N_BOOT_DEFAULT, seed=SEED_DEFAULT,
               power_const=MDE_CONST_80):
    """
    Minimum detectable effect for a paired comparison of a vs b.

    Returns a dict with the observed effect, the bootstrap standard error,
    the MDE at 80% power, the ratio of observed effect to MDE, and flags
    describing why a comparison is uninformative when it is.

    `informative` is False when no user can register a difference, i.e.
    every paired difference is exactly zero. In that case mde is None.
    """
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    n = len(a)
    diff = a - b
    observed = float(diff.mean()) if n else float("nan")

    n_nonzero_diff = int((diff != 0).sum())
    degenerate = n_nonzero_diff == 0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    se_boot = float(boot.std(ddof=1))

    mde = None if degenerate else power_const * se_boot
    return {
        "n": n,
        "a_mean": float(a.mean()) if n else float("nan"),
        "b_mean": float(b.mean()) if n else float("nan"),
        "observed_diff": observed,
        "se_boot": se_boot,
        "mde_80": mde,
        # |observed| / MDE: >= 1 means the observed effect is at or above the
        # detection threshold; << 1 means the test was never going to see it.
        "observed_over_mde": (abs(observed) / mde) if mde else None,
        "n_users_with_nonzero_diff": n_nonzero_diff,
        "n_users_a_nonzero": int((a != 0).sum()),
        "n_users_b_nonzero": int((b != 0).sum()),
        "informative": not degenerate,
    }


def summarise(rows, label_keys=("dataset", "stratum", "metric")):
    """Format a list of paired_mde results (each merged with its labels)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = [c for c in label_keys if c in df.columns] + [
        "n", "a_mean", "b_mean", "observed_diff", "se_boot", "mde_80",
        "observed_over_mde", "n_users_with_nonzero_diff", "informative",
    ]
    return df[[c for c in cols if c in df.columns]]
