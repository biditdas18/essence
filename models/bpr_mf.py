"""
models/bpr_mf.py
-------------------
Tier-2 Step 6: BPR-MF (Bayesian Personalized Ranking Matrix
Factorization; Rendle et al., UAI 2009), chosen over LightGCN for
implementation-time simplicity given tonight's time budget.

Standard formulation:
  - Latent factors P (n_users x d), Q (n_items x d), learned by SGD on
    the pairwise BPR loss: for each observed (user u, positive item i)
    pair, sample a random unobserved item j for that user, and update
    P[u], Q[i], Q[j] to increase score(u,i) - score(u,j), where
    score(u,i) = P[u] . Q[i] (+ optional item bias b[i]).
  - Loss per triple: -log(sigmoid(score(u,i) - score(u,j))) + L2 reg.

This is a genuine collaborative baseline: representations are learned
from cross-user interaction data (no item content used at any stage),
same "why this qualifies as CF" logic as the paper's existing ItemKNN
baseline.

Mandatory per tonight's Step 6 instruction: do NOT trust any real-dataset
number from this model before models/test_bpr_mf.py passes -- the MIND
routing bug earlier tonight was only caught by exactly this kind of
check run BEFORE trusting real numbers.
"""
import numpy as np


class BPRMF:
    def __init__(self, n_users, n_items, n_factors=32, lr=0.05, reg=0.01, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.1, size=(n_users, n_factors)).astype(np.float64)
        self.Q = rng.normal(0, 0.1, size=(n_items, n_factors)).astype(np.float64)
        self.b = np.zeros(n_items, dtype=np.float64)

    def score(self, u, i):
        return self.P[u] @ self.Q[i] + self.b[i]

    def score_all_items(self, u):
        return self.P[u] @ self.Q.T + self.b

    def fit(self, user_pos_items: dict, n_epochs=20, seed=42, verbose=False):
        """
        user_pos_items: {user_idx: list of positive item_idx}
        """
        rng = np.random.default_rng(seed)
        users_with_pos = [u for u, items in user_pos_items.items() if items]
        pos_sets = {u: set(items) for u, items in user_pos_items.items()}
        n_triples = sum(len(v) for v in user_pos_items.values())

        for epoch in range(n_epochs):
            total_loss = 0.0
            order = rng.permutation(users_with_pos)
            n_updates = 0
            for u in order:
                items = user_pos_items[u]
                for i in items:
                    # sample a negative not in this user's positive set
                    j = rng.integers(0, self.n_items)
                    tries = 0
                    while j in pos_sets[u] and tries < 10:
                        j = rng.integers(0, self.n_items)
                        tries += 1
                    x_ui = self.score(u, i)
                    x_uj = self.score(u, j)
                    x_uij = x_ui - x_uj
                    sig = 1.0 / (1.0 + np.exp(-x_uij))
                    grad = 1.0 - sig  # d(-log sigmoid(x))/dx = sig - 1, so update direction is (1-sig)

                    p_u = self.P[u].copy()
                    q_i = self.Q[i].copy()
                    q_j = self.Q[j].copy()

                    self.P[u] += self.lr * (grad * (q_i - q_j) - self.reg * p_u)
                    self.Q[i] += self.lr * (grad * p_u - self.reg * q_i)
                    self.Q[j] += self.lr * (grad * (-p_u) - self.reg * q_j)
                    self.b[i] += self.lr * (grad - self.reg * self.b[i])
                    self.b[j] += self.lr * (-grad - self.reg * self.b[j])

                    total_loss += -np.log(sig + 1e-12)
                    n_updates += 1
            if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
                print(f"    epoch {epoch+1}/{n_epochs}  mean BPR loss={total_loss/max(n_updates,1):.4f}")
        return self


def bpr_recommend(user_idx, seen_item_idx: set, model: BPRMF, item_ids: list, M=10):
    scores = model.score_all_items(user_idx).copy()
    for idx in seen_item_idx:
        scores[idx] = -np.inf
    top_idx = np.lexsort((np.arange(len(scores)), -scores))
    return [item_ids[i] for i in top_idx if scores[i] > -np.inf][:M]
