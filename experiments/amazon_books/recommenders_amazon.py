"""
Step 3 — Recommenders for Amazon Books experiment.

Three systems (exact logic from models/recommenders.py):
  A. CF Baseline      — global popularity, no embeddings
  B. Content Baseline — cosine similarity to mean user embedding
  C. Essence (K=3)    — cosine similarity to active K-means centroid

Run A: profile embeddings = metadata embeddings  (item_id -> vec)
       candidate embeddings = metadata embeddings (item_id -> vec)

Run B: profile embeddings = user-review embeddings ((user_id,item_id) -> vec)
       candidate embeddings = metadata embeddings  (item_id -> vec)
       (unseen items have no user review — use metadata for scoring)

CF baseline is identical in both runs (no embeddings used).
"""

import numpy as np
from sklearn.cluster import KMeans


# ─── Shared helper ───────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─── System A: CF Baseline (Popularity) ─────────────────────────────────────

def cf_recommend(user_id, train_df, M: int = 10):
    seen       = set(train_df[train_df["user_id"] == user_id]["item_id"])
    popularity = train_df.groupby("item_id").size()
    candidates = popularity[~popularity.index.isin(seen)]
    return candidates.nlargest(M).index.tolist()


# ─── System B: Content Baseline (Average Embedding) ─────────────────────────

def content_recommend(user_id, train_df,
                      profile_emb: dict,   # used to build user vector
                      candidate_emb: dict, # used to score unseen items
                      M: int = 10):
    """
    profile_emb  : {item_id: vec} for Run A
                   {(user_id, item_id): vec} for Run B
    candidate_emb: {item_id: vec} always
    """
    seen      = set(train_df[train_df["user_id"] == user_id]["item_id"])
    seen_vecs = _get_profile_vecs(user_id, seen, profile_emb)

    if not seen_vecs:
        return []

    user_vec   = np.mean(seen_vecs, axis=0)
    candidates = {iid: v for iid, v in candidate_emb.items() if iid not in seen}
    scores     = {iid: _cosine_sim(user_vec, v) for iid, v in candidates.items()}
    return sorted(scores, key=scores.get, reverse=True)[:M]


# ─── System C: Essence (K=3) ─────────────────────────────────────────────────

def essence_recommend(user_id, train_df,
                      profile_emb: dict,
                      candidate_emb: dict,
                      K: int = 3, M: int = 10):
    """
    profile_emb  : {item_id: vec} for Run A
                   {(user_id, item_id): vec} for Run B
    candidate_emb: {item_id: vec} always
    """
    user_rows = (
        train_df[train_df["user_id"] == user_id]
        .sort_values("timestamp", na_position="last")
    )
    seen = list(user_rows["item_id"])
    vecs = _get_profile_vecs(user_id, seen, profile_emb)

    if len(vecs) < K:
        return content_recommend(user_id, train_df, profile_emb, candidate_emb, M)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(np.array(vecs))

    # Active centroid: closest to mean of last 10 items
    recent      = seen[-10:]
    recent_vecs = _get_profile_vecs(user_id, recent, profile_emb)
    if recent_vecs:
        recent_mean    = np.mean(recent_vecs, axis=0)
        dists          = [np.linalg.norm(c - recent_mean) for c in km.cluster_centers_]
        active_centroid = km.cluster_centers_[np.argmin(dists)]
    else:
        active_centroid = km.cluster_centers_[0]

    seen_set   = set(seen)
    candidates = {iid: v for iid, v in candidate_emb.items() if iid not in seen_set}
    scores     = {iid: _cosine_sim(active_centroid, v) for iid, v in candidates.items()}
    return sorted(scores, key=scores.get, reverse=True)[:M]


# ─── Internal: resolve profile lookup for Run A vs Run B ─────────────────────

def _get_profile_vecs(user_id, item_ids, profile_emb: dict) -> list:
    """
    Handles two key schemas:
      Run A: profile_emb keys are item_id (str)
      Run B: profile_emb keys are (user_id, item_id) tuples
    """
    vecs = []
    # Detect key schema from a sample key
    sample_key = next(iter(profile_emb)) if profile_emb else None
    use_pair   = isinstance(sample_key, tuple)

    for iid in item_ids:
        key = (user_id, iid) if use_pair else iid
        v   = profile_emb.get(key)
        if v is not None:
            vecs.append(v)
    return vecs
