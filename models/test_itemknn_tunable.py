"""
models/test_itemknn_tunable.py
---------------------------------
Correctness check for TunableItemKNNModel, run BEFORE trusting any
real-dataset result from it (same discipline as
experiments/mind_comirec/test_mind_routing.py, applied here per Step 4's
own standard even though Step 4 only explicitly mandated a check for
Step 6's new baseline -- an untested new model implementation is an
untested new model implementation either way).

Two checks:
  1. Equivalence: TunableItemKNNModel(k_nn=None, shrinkage=0) must
     produce the SAME top-M recommendations, for every user, as the
     paper's existing canonical ItemKNNModel (models/recommenders.py) on
     a small synthetic dataset -- this is the mathematical claim the
     docstring makes, verified directly rather than assumed.
  2. Directional sanity: with shrinkage > 0, an item pair with only 1
     shared user must score lower (relative to the same raw cosine
     similarity) than a pair with many shared users -- shrinkage should
     visibly discount low-evidence similarity, not leave it unchanged.

Run:
    python models/test_itemknn_tunable.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend
from models.itemknn_tunable import TunableItemKNNModel, tunable_itemknn_recommend


def make_synthetic_df():
    # 6 users, 8 items, a mix of popular/co-occurring and rare/singleton items.
    rows = [
        ("u1", "i1"), ("u1", "i2"), ("u1", "i3"),
        ("u2", "i1"), ("u2", "i2"), ("u2", "i4"),
        ("u3", "i1"), ("u3", "i2"), ("u3", "i5"),
        ("u4", "i2"), ("u4", "i3"), ("u4", "i6"),
        ("u5", "i1"), ("u5", "i7"),
        ("u6", "i3"), ("u6", "i8"),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def check_equivalence():
    print("[1] Equivalence check: TunableItemKNNModel(k_nn=None, shrinkage=0) vs. canonical ItemKNNModel")
    df = make_synthetic_df()
    canonical = build_itemknn_model(df, item_col="item_id")
    tunable = TunableItemKNNModel(df, item_col="item_id", k_nn=None, shrinkage=0.0)

    all_users = sorted(df["user_id"].unique())
    mismatches = []
    for uid in all_users:
        rec_canonical = cf_itemknn_recommend(uid, df, canonical, M=10)
        rec_tunable = tunable_itemknn_recommend(uid, df, tunable, M=10)
        if rec_canonical != rec_tunable:
            mismatches.append((uid, rec_canonical, rec_tunable))

    if mismatches:
        print(f"  FAIL: {len(mismatches)}/{len(all_users)} users mismatched:")
        for uid, a, b in mismatches:
            print(f"    {uid}: canonical={a}  tunable={b}")
        return False
    print(f"  PASS: all {len(all_users)} users match exactly.")
    return True


def check_shrinkage_direction():
    print("\n[2] Directional sanity check: shrinkage discounts low-evidence similarity")
    # i1/i2 co-occur for u1,u2,u3 (3 shared users, high evidence).
    # i3/i8 in the synthetic set only co-occur for u6 (1 shared user) -- but
    # to isolate shrinkage's effect cleanly (not raw cosine differences too),
    # build a controlled pair: two item pairs with IDENTICAL raw cosine
    # similarity (1.0, both consumed by exactly the same set of users) but
    # different absolute overlap counts.
    rows_high = [(f"u{k}", "iA") for k in range(20)] + [(f"u{k}", "iB") for k in range(20)]
    rows_low = [("v1", "iC"), ("v1", "iD")]
    df = pd.DataFrame(rows_high + rows_low, columns=["user_id", "item_id"])

    m_shrunk = TunableItemKNNModel(df, item_col="item_id", k_nn=None, shrinkage=5.0)
    iA, iB = m_shrunk.item_idx["iA"], m_shrunk.item_idx["iB"]
    iC, iD = m_shrunk.item_idx["iC"], m_shrunk.item_idx["iD"]
    sim_high_evidence = m_shrunk.S[iA, iB]
    sim_low_evidence = m_shrunk.S[iC, iD]

    m_unshrunk = TunableItemKNNModel(df, item_col="item_id", k_nn=None, shrinkage=0.0)
    raw_high = m_unshrunk.S[iA, iB]
    raw_low = m_unshrunk.S[iC, iD]

    print(f"  Raw cosine similarity: high-evidence pair (20 shared users)={raw_high:.4f}, "
          f"low-evidence pair (1 shared user)={raw_low:.4f}")
    print(f"  Shrunk similarity (shrinkage=5): high-evidence={sim_high_evidence:.4f}, "
          f"low-evidence={sim_low_evidence:.4f}")

    expected_high = raw_high * 20 / (20 + 5)
    expected_low = raw_low * 1 / (1 + 5)
    ok = np.isclose(sim_high_evidence, expected_high, atol=1e-6) and \
         np.isclose(sim_low_evidence, expected_low, atol=1e-6) and \
         (sim_low_evidence / max(raw_low, 1e-9)) < (sim_high_evidence / max(raw_high, 1e-9))
    print(f"  Expected: high={expected_high:.4f}, low={expected_low:.4f}")
    print("  PASS: shrinkage formula matches exactly, low-evidence pair discounted proportionally more."
          if ok else "  FAIL")
    return ok


if __name__ == "__main__":
    ok1 = check_equivalence()
    ok2 = check_shrinkage_direction()
    print(f"\n{'='*50}")
    if ok1 and ok2:
        print("ALL CHECKS PASSED -- safe to trust TunableItemKNNModel on real data.")
    else:
        print("CHECKS FAILED -- do NOT trust any real-dataset result until fixed.")
        raise SystemExit(1)
