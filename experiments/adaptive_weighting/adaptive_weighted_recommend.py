"""
experiments/adaptive_weighting/adaptive_weighted_recommend.py
--------------------------------------------------------------------
Step 2: AdaptiveWeightedRecommend.

SCOPE, STATED PLAINLY (do not let this drift in any later reporting):
This is a scoped pilot test of a DIFFERENT hypothesis than what Essence's
paper tested. It is NOT audio-feature separation (Spotify's audio-features
API is deprecated for new apps as of Nov 2024 -- confirmed, no workaround,
blocked). It is NOT real user dislike/negative feedback (none of these
three datasets have explicit negative signal -- implicit feedback only,
same as everywhere else in this repo). This is a 2-CHANNEL (coarse, not
fine-grained) adaptive weighting test over EXISTING TEXT EMBEDDINGS, using
a PROXY for feedback (the user's own next chronological interaction, and a
randomly sampled non-interacted item as a pseudo-negative) -- not real
feedback of any kind. Never describe this as "audio features" or "user
feedback" in comments, logs, or downstream reporting.

Mechanism
---------
Per user, maintain a 2-dim weight vector (w_A, w_B), init (0.5, 0.5).
User profile per channel = recency-weighted mean of that channel's item
embeddings over the user's last N=10 train interactions (reuses the
existing recency_weighted_recommend query-construction logic, applied
separately per channel -- isolates the NEW variable, adaptive per-channel
weighting, from the recency mechanism already validated elsewhere in this
repo, rather than re-deriving both at once).

Score(candidate) = w_A * cos(candidate_A, profile_A) + w_B * cos(candidate_B, profile_B)

Online update, walking the user's train sequence chronologically:
  - At step t, score candidates using weights as of t-1.
  - Positive (proxy) signal: the actual next train item. Compute each
    channel's raw contribution to that item's score (w_A*cos_A vs
    w_B*cos_B before the update); nudge weight toward whichever channel
    contributed more, then renormalize to sum to 1.
  - Negative (proxy, explicitly disclosed) signal: one randomly sampled
    non-interacted item per step. If a channel's similarity to that
    pseudo-negative is high, nudge weight away from that channel slightly.
  - Fixed learning rate (default 0.05), NOT tuned -- flagged as an
    unoptimized default per the task's explicit instruction not to spend
    time on a sweep tonight.
"""

import numpy as np

LR_DEFAULT = 0.05
RECENCY_N = 10
RECENCY_DECAY = 0.9


def normalized(v):
    """L2-normalize. Row-wise for a 2D matrix, single-vector otherwise --
    np.linalg.norm(v) on a full matrix returns the Frobenius norm, not
    per-row norms, which would silently corrupt matrix scale."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 2:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-8)
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def recency_weighted_vec(vecs, decay=RECENCY_DECAY):
    """Same construction as recency_weighted_recommend's query, applied to one channel."""
    if not vecs:
        return None
    weights = np.array([decay ** i for i in range(len(vecs) - 1, -1, -1)])
    weights = weights / weights.sum()
    return np.average(np.array(vecs), axis=0, weights=weights)


def fit_adaptive_weights(train_seq, channel_a_map, channel_b_map, all_item_ids,
                         lr=LR_DEFAULT, recency_n=RECENCY_N, seed=42, disable_update=False):
    """
    disable_update: sanity-check mode (Step 1 of the follow-up session) --
    skips the online proxy-feedback walk entirely and returns the fixed
    (0.5, 0.5) prior. Used to confirm AdaptiveWeighted collapses to
    Recency-Weighted-like behavior when weighting is inert, before trusting
    any result from the learned-weight path.

    Walks train_seq (chronological list of item_ids) and returns the final
    (w_a, w_b) learned via the proxy-feedback online update described above.

    train_seq: list of item_ids, chronological, TRAIN ONLY (no test leakage).
    """
    rng = np.random.default_rng(seed)
    w = np.array([0.5, 0.5])

    if disable_update:
        return w

    valid_seq = [i for i in train_seq if i in channel_a_map and i in channel_b_map]
    if len(valid_seq) < 2:
        return w  # not enough signal to adapt; stays at the (0.5, 0.5) prior

    all_item_ids_arr = np.array(all_item_ids)

    for t in range(1, len(valid_seq)):
        history = valid_seq[max(0, t - recency_n):t]
        target = valid_seq[t]

        a_vecs = [channel_a_map[i] for i in history]
        b_vecs = [channel_b_map[i] for i in history]
        profile_a = recency_weighted_vec(a_vecs)
        profile_b = recency_weighted_vec(b_vecs)
        if profile_a is None or profile_b is None:
            continue

        # Positive (proxy) signal: the actual next train item
        pos_a = w[0] * cosine_sim(channel_a_map[target], profile_a)
        pos_b = w[1] * cosine_sim(channel_b_map[target], profile_b)
        if pos_a > pos_b:
            w[0] += lr
            w[1] -= lr
        elif pos_b > pos_a:
            w[1] += lr
            w[0] -= lr

        # Negative (proxy, disclosed) signal: one random non-interacted item
        seen_so_far = set(valid_seq[:t + 1])
        neg_candidates = all_item_ids_arr[~np.isin(all_item_ids_arr, list(seen_so_far))]
        if len(neg_candidates) > 0:
            neg_item = rng.choice(neg_candidates)
            if neg_item in channel_a_map and neg_item in channel_b_map:
                neg_a = cosine_sim(channel_a_map[neg_item], profile_a)
                neg_b = cosine_sim(channel_b_map[neg_item], profile_b)
                if neg_a > neg_b:
                    w[0] -= lr * 0.5  # smaller nudge for the negative signal (proxy, noisier than the positive)
                elif neg_b > neg_a:
                    w[1] -= lr * 0.5

        w = np.clip(w, 0.01, 0.99)
        w = w / w.sum()

    return w


def adaptive_weighted_recommend(user_id, train_items, channel_a_map, channel_b_map,
                                candidate_item_ids, candidate_a_matrix, candidate_b_matrix,
                                seen_mask, M=10, lr=LR_DEFAULT, recency_n=RECENCY_N, seed=42,
                                disable_update=False):
    """
    Full pipeline for one user: learn (w_a, w_b) via the proxy-feedback walk
    over train_items, then score all unseen candidates and return top-M.

    candidate_a_matrix / candidate_b_matrix: (n_items, dim) L2-normalized,
    row order matching candidate_item_ids -- same convention as the
    vectorized top_k_unseen pattern used throughout this repo.

    disable_update: sanity-check mode, see fit_adaptive_weights.
    """
    w = fit_adaptive_weights(train_items, channel_a_map, channel_b_map,
                             candidate_item_ids, lr=lr, recency_n=recency_n, seed=seed,
                             disable_update=disable_update)

    recent = [i for i in train_items[-recency_n:] if i in channel_a_map and i in channel_b_map]
    if not recent:
        return [], w
    profile_a = normalized(recency_weighted_vec([channel_a_map[i] for i in recent]))
    profile_b = normalized(recency_weighted_vec([channel_b_map[i] for i in recent]))

    scores = w[0] * (candidate_a_matrix @ profile_a) + w[1] * (candidate_b_matrix @ profile_b)
    scores = np.asarray(scores).copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:M]
    recs = [candidate_item_ids[i] for i in top_idx]
    return recs, w
