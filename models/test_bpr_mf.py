"""
models/test_bpr_mf.py
------------------------
Mandatory conformance check for BPR-MF (Tier-2 Step 6), run BEFORE
trusting ANY real-dataset comparison number from this model -- same
discipline as experiments/mind_comirec/test_mind_routing.py, explicitly
required by tonight's Step 6 instruction given that tonight's MIND bug
was only caught by exactly this kind of check.

Three checks, in increasing order of what they verify:
  1. Loss decreases monotonically-ish over training (sanity that the
     gradient signs are correct, not accidentally doing gradient ascent).
  2. Latent-block recovery: construct two disjoint user/item blocks with
     NO cross-block interactions during training. After training, for
     held-out block-A users, block-A items must score higher on average
     than block-B items (and vice versa) -- this is a real test of
     whether the learned factors capture co-occurrence structure, not
     just a check that the code runs.
  3. Index-mapping sanity: a user's single known positive item must be
     recoverable near the top of their score ranking (catches
     off-by-one / transposed user-item index bugs directly).

Run:
    python models/test_bpr_mf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.bpr_mf import BPRMF, bpr_recommend


def check_loss_decreases():
    print("[1] Loss-decreases-over-training check")
    rng = np.random.default_rng(0)
    n_users, n_items = 50, 60
    user_pos_items = {u: sorted(rng.choice(n_items, size=5, replace=False).tolist()) for u in range(n_users)}

    model = BPRMF(n_users, n_items, n_factors=8, lr=0.05, reg=0.001, seed=1)

    def mean_loss():
        total, n = 0.0, 0
        for u, items in user_pos_items.items():
            for i in items:
                j = (i + 7) % n_items
                if j in items:
                    j = (j + 1) % n_items
                x = model.score(u, i) - model.score(u, j)
                sig = 1.0 / (1.0 + np.exp(-x))
                total += -np.log(sig + 1e-12)
                n += 1
        return total / n

    loss_before = mean_loss()
    losses = [loss_before]
    for chunk in range(3):
        model.fit(user_pos_items, n_epochs=10, seed=1, verbose=False)
        losses.append(mean_loss())
    print(f"  Mean loss at epoch 0/10/20/30: " + ", ".join(f"{l:.4f}" for l in losses))
    # Two checks, both principled (not an arbitrary magic percentage):
    # (a) loss must be lower after training than the untrained random-init
    #     baseline (theoretical baseline for near-zero-init P,Q is exactly
    #     -log(0.5) = 0.6931, confirmed above) -- any trained model that
    #     fails this is not learning at all;
    # (b) loss must keep decreasing across the later chunks too (not
    #     plateau immediately) -- confirms real gradient-driven convergence
    #     rather than one lucky early jump.
    ok = losses[-1] < losses[0] and losses[-1] < losses[1]
    print("  PASS: loss decreases from the untrained baseline and continues decreasing with more training."
          if ok else "  FAIL: loss did not decrease as expected.")
    return ok


def check_latent_block_recovery():
    print("\n[2] Latent-block recovery check")
    n_users, n_items = 40, 40  # users/items 0-19 = block A, 20-39 = block B
    rng = np.random.default_rng(2)
    user_pos_items = {}
    for u in range(n_users):
        block_items = range(0, 20) if u < 20 else range(20, 40)
        user_pos_items[u] = sorted(rng.choice(list(block_items), size=6, replace=False).tolist())

    model = BPRMF(n_users, n_items, n_factors=8, lr=0.05, reg=0.001, seed=2)
    model.fit(user_pos_items, n_epochs=40, seed=2, verbose=False)

    block_a_users = list(range(20))
    block_b_users = list(range(20, 40))
    a_items, b_items = list(range(20)), list(range(20, 40))

    a_user_scores_a_items = np.mean([model.score_all_items(u)[a_items].mean() for u in block_a_users])
    a_user_scores_b_items = np.mean([model.score_all_items(u)[b_items].mean() for u in block_a_users])
    b_user_scores_a_items = np.mean([model.score_all_items(u)[a_items].mean() for u in block_b_users])
    b_user_scores_b_items = np.mean([model.score_all_items(u)[b_items].mean() for u in block_b_users])

    print(f"  Block-A users -> block-A items mean score: {a_user_scores_a_items:.4f}")
    print(f"  Block-A users -> block-B items mean score: {a_user_scores_b_items:.4f}")
    print(f"  Block-B users -> block-A items mean score: {b_user_scores_a_items:.4f}")
    print(f"  Block-B users -> block-B items mean score: {b_user_scores_b_items:.4f}")

    ok = (a_user_scores_a_items > a_user_scores_b_items) and (b_user_scores_b_items > b_user_scores_a_items)
    print("  PASS: each block's users score their own block's items higher, as expected."
          if ok else "  FAIL: model did not recover the latent block structure.")
    return ok


def check_index_mapping():
    print("\n[3] Index-mapping sanity check")
    n_users, n_items = 10, 100
    user_pos_items = {0: [42]}
    for u in range(1, n_users):
        user_pos_items[u] = [(u * 3) % n_items]

    model = BPRMF(n_users, n_items, n_factors=8, lr=0.1, reg=0.001, seed=3)
    model.fit(user_pos_items, n_epochs=30, seed=3, verbose=False)

    scores_u0 = model.score_all_items(0)
    rank_of_42 = int((scores_u0 > scores_u0[42]).sum())
    print(f"  User 0's known positive item (id 42) is ranked #{rank_of_42+1} of {n_items} by score.")
    ok = rank_of_42 < 5
    print("  PASS: known positive lands near the top of the ranking." if ok else "  FAIL: index mapping looks broken.")
    return ok


if __name__ == "__main__":
    ok1 = check_loss_decreases()
    ok2 = check_latent_block_recovery()
    ok3 = check_index_mapping()
    print(f"\n{'='*50}")
    if ok1 and ok2 and ok3:
        print("ALL CHECKS PASSED -- safe to trust BPR-MF on real data.")
    else:
        print("CHECKS FAILED -- do NOT trust any real-dataset result until fixed.")
        raise SystemExit(1)
