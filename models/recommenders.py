"""
models/recommenders.py
-----------------------
Five recommendation systems:

  A. Popularity Baseline  — non-personalized global popularity ranking
                            (renamed from CF Baseline per reviewer feedback:
                            this is not collaborative filtering)
  B. CF Baseline (ItemKNN)— proper item-based collaborative filtering using
                            cosine similarity over the user-item interaction
                            matrix; no embeddings, no cross-user training
  C. Content Baseline     — cosine similarity to mean user embedding
  D. Essence              — cosine similarity to active K-means centroid

All functions accept a pre-loaded train_df (and, where needed, a precomputed
model object) so they can be called from the evaluation loop without
re-loading data.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ---------------------------------------------------------------------------
# System A: Popularity Baseline (non-personalized)
# ---------------------------------------------------------------------------

def popularity_recommend(user_id, train_df, M: int = 10,
                         item_embedding_map: dict = None):
    """
    Recommends the M most globally popular tracks the user has not already
    heard in the train set.

    NOTE: This is a NON-PERSONALIZED popularity baseline, NOT collaborative
    filtering. Items not present in the training set have popularity 0 and
    are never recommended (nlargest(M) will always prefer any positive-count
    train item); output is therefore identical whether or not item_embedding_map
    is provided. The parameter is accepted for API consistency with the unified
    candidate-pool interface.

    Parameters
    ----------
    user_id            : user identifier (must exist in train_df)
    train_df           : full training interactions DataFrame
    M                  : number of recommendations to return
    item_embedding_map : accepted for interface consistency; does not change
                         output (test-only items have popularity 0).

    Returns
    -------
    list of track_id strings, length <= M
    """
    seen = set(train_df[train_df["user_id"] == user_id]["track_id"])
    popularity = train_df.groupby("track_id").size()
    candidates = popularity[~popularity.index.isin(seen)]
    return candidates.nlargest(M).index.tolist()


# Keep the old name as an alias so existing callers don't break during transition
cf_recommend = popularity_recommend


# ---------------------------------------------------------------------------
# System 0: Random Baseline
# ---------------------------------------------------------------------------

def random_recommend(user_id, train_df, M: int = 10, seed: int = 42,
                     item_embedding_map: dict = None):
    """
    Recommends M items chosen uniformly at random from items the user
    has not seen in the train set.

    Expected Recall@10 under random = M / |catalog|.

    Parameters
    ----------
    user_id            : user identifier
    train_df           : full training interactions DataFrame
    M                  : number of recommendations to return
    seed               : random seed (default 42; pass a stable per-user seed,
                         e.g. via hashlib.md5, to vary per user reproducibly)
    item_embedding_map : if provided, the full item universe (train∪test) is
                         used as the candidate pool; otherwise falls back to
                         train-only items. Pass this to ensure a unified
                         candidate pool across all five systems.

    Returns
    -------
    list of track_id strings, length <= M
    """
    seen = set(train_df[train_df["user_id"] == user_id]["track_id"])
    if item_embedding_map is not None:
        all_items = sorted(set(item_embedding_map.keys()) - seen)
    else:
        all_items = sorted(set(train_df["track_id"].unique()) - seen)
    rng = np.random.default_rng(seed)
    n   = min(M, len(all_items))
    return rng.choice(all_items, size=n, replace=False).tolist()


# ---------------------------------------------------------------------------
# System B: CF Baseline — Item-based KNN (real collaborative filtering)
# ---------------------------------------------------------------------------

class ItemKNNModel:
    """
    Precomputed Item-based KNN model.

    Algorithm
    ---------
    1. Build binary user-item matrix R of shape (n_users × n_items).
    2. L2-normalise each item's user-interaction vector (column of R):
         R_norm = normalise(R.T, norm='l2')   →  shape (n_items × n_users)
    3. At prediction time for user u with seen items S:
         query   = Σ_{j ∈ S} R_norm[j]        →  shape (n_users,)
         scores  = R_norm @ query              →  shape (n_items,)
       This equals Σ_{j ∈ S} cos_sim(item_i, item_j) for each candidate i,
       which is standard ItemKNN aggregate scoring.

    Complexity per user: O(|S| × n_users + n_items × n_users).
    With n_users=99 and n_items=22 767 this is trivially fast.

    Why this qualifies as CF
    ------------------------
    - Personalised: each user gets different recommendations.
    - Cross-user signal: item similarity is derived from which *other* users
      co-interacted with the same items (the user-item co-occurrence matrix).
    - No item content used at any stage.
    - Standard reference: Sarwar et al., "Item-Based Collaborative Filtering
      Recommendation Algorithms," WWW 2001.
    """

    def __init__(self, train_df: "pd.DataFrame", item_col: str = "track_id"):
        """
        Parameters
        ----------
        train_df : training interactions DataFrame
        item_col : name of the item ID column (default "track_id" for Last.fm;
                   use "item_id" for Amazon Books)
        """
        from scipy.sparse import csr_matrix

        self.item_col  = item_col

        # Index users and items
        self.all_users = sorted(train_df["user_id"].unique())
        self.all_items = sorted(train_df[item_col].unique())
        self.user_idx  = {u: i for i, u in enumerate(self.all_users)}
        self.item_idx  = {t: i for i, t in enumerate(self.all_items)}

        n_u = len(self.all_users)
        n_i = len(self.all_items)

        # Build binary user-item matrix as sparse (handles both small and large datasets)
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

        # L2-normalise each item vector (column of R → row of R.T)
        # R_norm shape: (n_items × n_users)
        self.R_norm = normalize(R.T, norm="l2")   # sparse (n_items, n_users)


def build_itemknn_model(train_df, item_col: str = "track_id") -> ItemKNNModel:
    """Build and return a precomputed ItemKNN model from train_df.

    Parameters
    ----------
    train_df : training interactions DataFrame
    item_col : item ID column name (default "track_id"; use "item_id" for Amazon)
    """
    return ItemKNNModel(train_df, item_col=item_col)


def cf_itemknn_recommend(user_id, train_df, itemknn_model: ItemKNNModel,
                         M: int = 10, item_embedding_map: dict = None):
    """
    Item-based KNN collaborative filtering recommendation.

    Parameters
    ----------
    user_id            : user identifier (must exist in train_df)
    train_df           : full training interactions DataFrame
    itemknn_model      : precomputed ItemKNNModel (call build_itemknn_model once)
    M                  : number of recommendations to return
    item_embedding_map : if provided, test-only items (in full catalog but absent
                         from the training matrix) are appended with score 0 to
                         fill any remaining slots after the ItemKNN-scored train
                         items. In practice this only activates when fewer than M
                         train items have positive ItemKNN scores (rare).

    Returns
    -------
    list of track_id strings, length <= M
    """
    model = itemknn_model
    seen  = set(train_df[train_df["user_id"] == user_id][model.item_col])

    # Indices of seen items that exist in the model's item set
    seen_indices = [model.item_idx[t] for t in seen if t in model.item_idx]
    if not seen_indices:
        return []

    # Aggregate query vector in user space
    # .sum(axis=0) on a sparse matrix returns (1, n_users); flatten to 1D array
    query  = np.asarray(model.R_norm[seen_indices].sum(axis=0)).ravel()  # (n_users,)

    # Score all training items — result is (n_items,)
    scores = np.asarray(model.R_norm @ query).ravel()

    # Mask seen items
    for idx in seen_indices:
        scores[idx] = -np.inf

    # Full stable sort (score desc, tie-break by index asc) — deterministic even
    # when argpartition would produce ambiguous boundary sets on tied scores.
    all_indices = np.lexsort((np.arange(len(scores)), -scores))
    top_indices = all_indices[:M]
    recs = [model.all_items[i] for i in top_indices if scores[i] > -np.inf][:M]

    # Fill remaining slots with test-only items (score 0) if full catalog provided.
    # These items have no co-occurrence signal and score below all train items
    # with any positive score, so this branch almost never activates in practice.
    if item_embedding_map is not None and len(recs) < M:
        train_item_set = set(model.all_items)
        test_only = [
            iid for iid in item_embedding_map
            if iid not in train_item_set and iid not in seen
        ]
        recs = (recs + test_only)[:M]

    return recs


# ---------------------------------------------------------------------------
# System C: Content Baseline (Average Embedding)
# ---------------------------------------------------------------------------

def content_recommend(user_id, train_df, item_embedding_map: dict, M: int = 10):
    """
    Computes mean of all user item embeddings.
    Returns M unseen items by cosine similarity to the mean vector.

    Parameters
    ----------
    user_id            : user identifier
    train_df           : full training interactions DataFrame
    item_embedding_map : dict {track_id: np.ndarray}
    M                  : number of recommendations to return

    Returns
    -------
    list of track_id strings, length <= M
    """
    seen = set(train_df[train_df["user_id"] == user_id]["track_id"])
    seen_vecs = [item_embedding_map[i] for i in seen if i in item_embedding_map]

    if len(seen_vecs) == 0:
        return []

    user_vec = np.mean(seen_vecs, axis=0)

    # Score all unseen items
    candidates = {
        tid: emb
        for tid, emb in item_embedding_map.items()
        if tid not in seen
    }

    scores = {
        tid: cosine_similarity(user_vec, emb)
        for tid, emb in candidates.items()
    }

    return sorted(scores, key=scores.get, reverse=True)[:M]


# ---------------------------------------------------------------------------
# System D: Essence
# ---------------------------------------------------------------------------

def essence_recommend(user_id, train_df, item_embedding_map: dict,
                      K: int = 3, M: int = 10):
    """
    Clusters user history into K centroids via K-means.
    Selects active centroid by proximity to mean of the last 10 items
    chronologically in the train set.
    Returns M unseen items by cosine similarity to the active centroid.

    NOTE on "last 10 items": this refers to the chronologically last 10
    items in the user's TRAIN set (sorted by timestamp). Under a
    chronological 80/20 split, these are the user's genuinely most recent
    interactions before the test period — consistent with Section 3.5's
    description of active-cluster selection by recent context.

    Fallback: if the user has fewer than K embeddable items,
              delegates to content_recommend.

    Parameters
    ----------
    user_id            : user identifier
    train_df           : full training interactions DataFrame
    item_embedding_map : dict {track_id: np.ndarray}
    K                  : number of K-means clusters
    M                  : number of recommendations to return

    Returns
    -------
    list of track_id strings, length <= M
    """
    # Retrieve user history in chronological order (timestamp ascending)
    user_rows = (
        train_df[train_df["user_id"] == user_id]
        .sort_values("timestamp")
    )
    seen = list(user_rows["track_id"])
    vecs = [item_embedding_map[i] for i in seen if i in item_embedding_map]

    # Fallback guard: need at least K items to cluster
    if len(vecs) < K:
        return content_recommend(user_id, train_df, item_embedding_map, M)

    # Fit K-means on full user history
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(np.array(vecs))

    # Active centroid: closest to mean of last 10 chronological train items
    recent = seen[-10:]
    recent_vecs = [item_embedding_map[i] for i in recent if i in item_embedding_map]

    if len(recent_vecs) == 0:
        active_centroid = km.cluster_centers_[0]
    else:
        recent_mean = np.mean(recent_vecs, axis=0)
        dists = [np.linalg.norm(c - recent_mean) for c in km.cluster_centers_]
        active_centroid = km.cluster_centers_[np.argmin(dists)]

    # Score all unseen items against the active centroid
    seen_set = set(seen)
    candidates = {
        tid: emb
        for tid, emb in item_embedding_map.items()
        if tid not in seen_set
    }

    scores = {
        tid: cosine_similarity(active_centroid, emb)
        for tid, emb in candidates.items()
    }

    return sorted(scores, key=scores.get, reverse=True)[:M]


# ---------------------------------------------------------------------------
# Ablation variant: Essence with mean-of-all-train selection (not last-10)
# ---------------------------------------------------------------------------

def essence_recommend_meanselect(user_id, train_df, item_embedding_map: dict,
                                  K: int = 3, M: int = 10):
    """
    Ablation variant of Essence for diagnostic purposes ONLY.

    Identical to essence_recommend() in every respect — K-means clustering,
    cosine retrieval, fallback logic — EXCEPT the active centroid is chosen
    as the centroid closest to the MEAN of the user's ENTIRE train embedding,
    rather than the mean of the chronologically last 10 items.

    Purpose: isolates whether the recency heuristic (last-10) hurts or helps
    LT-Recall on small/sparse datasets (Last.fm-1K) vs. using the full-history
    mean as the cluster-selection signal.

    DO NOT use in production or final evaluation. Kept alongside the canonical
    essence_recommend() for ablation comparison only.

    Parameters
    ----------
    user_id            : user identifier
    train_df           : full training interactions DataFrame
    item_embedding_map : dict {track_id: np.ndarray}
    K                  : number of K-means clusters (default 3)
    M                  : number of recommendations to return

    Returns
    -------
    list of track_id strings, length <= M
    """
    user_rows = (
        train_df[train_df["user_id"] == user_id]
        .sort_values("timestamp")
    )
    seen = list(user_rows["track_id"])
    vecs = [item_embedding_map[i] for i in seen if i in item_embedding_map]

    if len(vecs) < K:
        return content_recommend(user_id, train_df, item_embedding_map, M)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(np.array(vecs))

    # Active centroid: closest to mean of FULL train history (ablation change)
    full_mean = np.mean(vecs, axis=0)
    dists = [np.linalg.norm(c - full_mean) for c in km.cluster_centers_]
    active_centroid = km.cluster_centers_[np.argmin(dists)]

    seen_set = set(seen)
    candidates = {
        tid: emb
        for tid, emb in item_embedding_map.items()
        if tid not in seen_set
    }

    scores = {
        tid: cosine_similarity(active_centroid, emb)
        for tid, emb in candidates.items()
    }

    return sorted(scores, key=scores.get, reverse=True)[:M]
