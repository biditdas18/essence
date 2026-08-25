"""
models/itemknn_tunable.py
---------------------------
Tier-2 Step 4: a tunable variant of ItemKNN (neighbor count K_nn, shrinkage
term) built alongside the existing untuned full-neighborhood ItemKNNModel
in models/recommenders.py (never modified -- that model stays the paper's
canonical, untuned baseline).

Standard item-based CF with top-K neighbor truncation and shrinkage
regularization (Sarwar et al. 2001; shrinkage per Bell & Koren 2007):

    raw_sim(i, j)    = cosine similarity of items i, j over the binary
                        user-item matrix
    co(i, j)         = number of users who interacted with both i and j
    shrunk_sim(i, j) = raw_sim(i, j) * co(i, j) / (co(i, j) + shrinkage)
    neighbors(i)     = the K_nn items j with the highest shrunk_sim(i, j)
                        (co(i,j) > 0 required -- items with zero
                        co-occurrence are never neighbors regardless of
                        shrinkage)
    score(u, i)      = sum of shrunk_sim(i, j) over j in seen(u) that are
                        also in neighbors(i)  (symmetric neighbor check:
                        i must be among j's top-K too, standard
                        item-based-KNN convention -- avoids one-sided
                        neighbor inflation)

Implemented with sparse matrix operations throughout (R_norm @ R_norm.T
computed as a SPARSE product -- co-occurrence in real interaction data is
naturally sparse, so this never materializes a dense n_items x n_items
matrix, which would be infeasible at Amazon's 61,727-item scale).

shrinkage=0 and K_nn=None (no truncation) is mathematically equivalent to
the paper's existing canonical ItemKNNModel (models/recommenders.py) --
this equivalence is the correctness check verified in
verify_itemknn_tunable.py before any real dataset is touched.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags


class TunableItemKNNModel:
    def __init__(self, train_df, item_col="item_id", k_nn=None, shrinkage=0.0):
        """
        Parameters
        ----------
        k_nn      : int or None. Max neighbors kept per item (None = no
                    truncation, i.e. full neighborhood).
        shrinkage : float >= 0. Shrinkage constant (0 = no shrinkage).
        """
        self.item_col = item_col
        self.k_nn = k_nn
        self.shrinkage = shrinkage

        self.all_users = sorted(train_df["user_id"].unique())
        self.all_items = sorted(train_df[item_col].unique())
        self.user_idx = {u: i for i, u in enumerate(self.all_users)}
        self.item_idx = {t: i for i, t in enumerate(self.all_items)}
        n_u, n_i = len(self.all_users), len(self.all_items)

        rows_idx, cols_idx = [], []
        for _, row in train_df.iterrows():
            ui = self.user_idx.get(row["user_id"])
            ii = self.item_idx.get(row[item_col])
            if ui is not None and ii is not None:
                rows_idx.append(ui)
                cols_idx.append(ii)

        R = csr_matrix(
            (np.ones(len(rows_idx), dtype=np.float32), (rows_idx, cols_idx)),
            shape=(n_u, n_i),
        )
        self.R = R  # (n_users, n_items) binary

        # Co-occurrence counts: C[i,j] = # users who touched both i and j
        C = (R.T @ R).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        # Cosine similarity via L2-normalized item columns
        norms = np.sqrt(np.asarray(R.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        Dinv = diags(1.0 / norms)
        R_normed_items = (R @ Dinv).tocsr()  # each item column now L2-unit
        S_raw = (R_normed_items.T @ R_normed_items).tocsr()  # cosine sim, sparse
        S_raw.setdiag(0)
        S_raw.eliminate_zeros()

        if shrinkage > 0:
            # shrunk = raw * co / (co + shrinkage), elementwise on the shared sparsity pattern
            C_aligned = C.multiply(S_raw.astype(bool))  # co counts at S_raw's nonzero positions
            denom = C_aligned.copy()
            denom.data = denom.data + shrinkage
            factor = C_aligned.copy()
            factor.data = factor.data / denom.data
            S = S_raw.multiply(factor).tocsr()
        else:
            S = S_raw

        if k_nn is not None:
            S = _row_top_k(S, k_nn)

        self.S = S  # (n_items, n_items) sparse similarity/neighbor matrix


def _row_top_k(mat_csr, k):
    """Keep only the top-k largest entries per row of a sparse CSR matrix."""
    mat = mat_csr.tocsr().copy()
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if end - start > k:
            row_data = mat.data[start:end]
            # indices of the (end-start-k) smallest values -> zero them out
            keep_idx = np.argpartition(row_data, -k)[-k:]
            mask = np.ones(end - start, dtype=bool)
            mask[keep_idx] = False
            row_data[mask] = 0.0
    mat.eliminate_zeros()
    return mat


def tunable_itemknn_recommend(user_id, train_df, model: TunableItemKNNModel, M=10):
    seen = set(train_df[train_df["user_id"] == user_id][model.item_col])
    seen_indices = [model.item_idx[t] for t in seen if t in model.item_idx]
    if not seen_indices:
        return []
    # score(candidate) = sum over seen items of S[seen, candidate]
    scores = np.asarray(model.S[seen_indices, :].sum(axis=0)).ravel()
    seen_mask = np.zeros(len(model.all_items), dtype=bool)
    seen_mask[seen_indices] = True
    scores[seen_mask] = -np.inf
    all_idx = np.lexsort((np.arange(len(scores)), -scores))
    return [model.all_items[i] for i in all_idx if scores[i] > -np.inf][:M]
