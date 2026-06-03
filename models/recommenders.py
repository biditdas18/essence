"""
models/recommenders.py
-----------------------
Three recommendation systems:

  A. CF Baseline         — global popularity ranking
  B. Content Baseline    — cosine similarity to mean user embedding
  C. Essence             — cosine similarity to active K-means centroid

All functions accept a pre-loaded train_df and item_embedding_map
so they can be called from the evaluation loop without re-loading data.
"""

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ---------------------------------------------------------------------------
# System A: CF Baseline (Popularity)
# ---------------------------------------------------------------------------

def cf_recommend(user_id, train_df, M: int = 10):
    """
    Recommends the M most globally popular tracks
    the user has not already heard in the train set.

    Parameters
    ----------
    user_id   : user identifier (must exist in train_df)
    train_df  : full training interactions DataFrame
    M         : number of recommendations to return

    Returns
    -------
    list of track_id strings, length <= M
    """
    seen = set(train_df[train_df["user_id"] == user_id]["track_id"])
    popularity = train_df.groupby("track_id").size()
    candidates = popularity[~popularity.index.isin(seen)]
    return candidates.nlargest(M).index.tolist()


# ---------------------------------------------------------------------------
# System B: Content Baseline (Average Embedding)
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
# System C: Essence
# ---------------------------------------------------------------------------

def essence_recommend(user_id, train_df, item_embedding_map: dict,
                      K: int = 3, M: int = 10):
    """
    Clusters user history into K centroids via K-means.
    Selects active centroid by proximity to mean of the last 10 items.
    Returns M unseen items by cosine similarity to the active centroid.

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
    # Retrieve user history in chronological order
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

    # Active centroid: closest to mean of last 10 items
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
