"""
experiments/adaptive_weighting/combined_cluster_adaptive.py
--------------------------------------------------------------------
Step 4: combined clustering + adaptive weighting -- the design closest to
the original full thesis (cluster to find the interest zone, then adapt
per-channel weighting within that zone, not globally).

Mechanism
---------
1. Cluster the user's train history into K=3 via KMeans on the EXISTING
   single-vector metadata embedding (emb_meta) -- unchanged, reused
   directly from essence_recommend's own clustering logic, not a new
   joint-channel clustering space.
2. Active-cluster selection: unchanged, reused directly from
   essence_recommend (closest centroid to the mean of the last 10 items
   in emb_meta space).
3. NEW: per-CLUSTER adaptive weights (w_A^k, w_B^k), one pair per cluster
   k=0..K-1, each initialized (0.5, 0.5) and updated ONLY from training
   steps whose target item belongs to that cluster -- this is what makes
   the weighting "adapt within the currently-active interest cluster, not
   globally" rather than a single global (w_A, w_B) shared across a
   user's whole history. Uses the same z-scored proxy-feedback update
   from adaptive_weighted_recommend.py's fit_adaptive_weights (same fix,
   same disclosed proxy-signal caveats), just partitioned by cluster.
4. At recommendation time: use the active cluster's own (w_A^k, w_B^k)
   and its own channel-A/channel-B profile (recency-weighted mean of
   THAT cluster's member items only, not the global last-10 window) to
   score all unseen candidates.
"""

import numpy as np
from sklearn.cluster import KMeans

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from adaptive_weighted_recommend import normalized, cosine_sim, recency_weighted_vec, N_REF_SAMPLE

LR_DEFAULT = 0.05
RECENCY_N = 10
K = 3


def fit_cluster_adaptive_weights(train_seq, cluster_of_item, channel_a_map, channel_b_map,
                                 all_item_ids, lr=LR_DEFAULT, recency_n=RECENCY_N, seed=42):
    """
    Same proxy-feedback z-scored update as fit_adaptive_weights, but
    partitioned into K independent (w_A^k, w_B^k) pairs, one per cluster,
    each updated only from steps whose target belongs to that cluster.
    Within-cluster profile is the recency-weighted mean of that cluster's
    OWN member items seen so far (not the global last-10 window).
    """
    rng = np.random.default_rng(seed)
    weights = {k: np.array([0.5, 0.5]) for k in range(K)}

    valid_seq = [i for i in train_seq if i in channel_a_map and i in channel_b_map and i in cluster_of_item]
    if len(valid_seq) < 2:
        return weights

    all_item_ids_arr = np.array(all_item_ids)
    cluster_history = {k: [] for k in range(K)}  # items seen so far, per cluster, in order

    def ref_stats(ref_items, profile_a, profile_b):
        ref_a = np.array([cosine_sim(channel_a_map[i], profile_a) for i in ref_items])
        ref_b = np.array([cosine_sim(channel_b_map[i], profile_b) for i in ref_items])
        return (ref_a.mean(), ref_a.std() + 1e-8), (ref_b.mean(), ref_b.std() + 1e-8)

    def zscore(value, mean_std):
        mean, std = mean_std
        return (value - mean) / std

    for t in range(len(valid_seq)):
        target = valid_seq[t]
        k = cluster_of_item[target]

        hist_k = cluster_history[k][-recency_n:]
        if len(hist_k) < 1:
            cluster_history[k].append(target)
            continue

        profile_a = recency_weighted_vec([channel_a_map[i] for i in hist_k])
        profile_b = recency_weighted_vec([channel_b_map[i] for i in hist_k])

        seen_so_far = set(valid_seq[:t + 1])
        unseen_pool = all_item_ids_arr[~np.isin(all_item_ids_arr, list(seen_so_far))]
        if len(unseen_pool) >= N_REF_SAMPLE:
            ref_items = rng.choice(unseen_pool, size=N_REF_SAMPLE, replace=False)
            ref_items = [i for i in ref_items if i in channel_a_map and i in channel_b_map]
            if len(ref_items) >= 5:
                stats_a, stats_b = ref_stats(ref_items, profile_a, profile_b)

                z_pos_a = zscore(cosine_sim(channel_a_map[target], profile_a), stats_a)
                z_pos_b = zscore(cosine_sim(channel_b_map[target], profile_b), stats_b)
                w = weights[k]
                if z_pos_a > z_pos_b:
                    w[0] += lr; w[1] -= lr
                elif z_pos_b > z_pos_a:
                    w[1] += lr; w[0] -= lr

                neg_item = rng.choice(ref_items)
                z_neg_a = zscore(cosine_sim(channel_a_map[neg_item], profile_a), stats_a)
                z_neg_b = zscore(cosine_sim(channel_b_map[neg_item], profile_b), stats_b)
                if z_neg_a > z_neg_b:
                    w[0] -= lr * 0.5
                elif z_neg_b > z_neg_a:
                    w[1] -= lr * 0.5

                w = np.clip(w, 0.01, 0.99)
                weights[k] = w / w.sum()

        cluster_history[k].append(target)

    return weights


def combined_cluster_adaptive_recommend(train_items, emb_meta, channel_a_map, channel_b_map,
                                        candidate_item_ids, candidate_a_matrix, candidate_b_matrix,
                                        seen_mask, M=10, K=K, lr=LR_DEFAULT, seed=42):
    """
    Full pipeline for one user. Returns (recs, active_cluster_id, learned_weights_dict).
    Falls back to an empty rec list if the user has fewer than K embeddable items
    (matching essence_recommend's own fallback threshold).
    """
    seen_vecs_meta = [emb_meta[i] for i in train_items if i in emb_meta]
    if len(seen_vecs_meta) < K:
        return [], None, {}

    valid_train_items = [i for i in train_items if i in emb_meta]
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(np.array(seen_vecs_meta))
    cluster_of_item = {iid: int(label) for iid, label in zip(valid_train_items, km.labels_)}

    # Active cluster selection: unchanged from essence_recommend
    recent_vecs = seen_vecs_meta[-10:]
    recent_mean = np.mean(recent_vecs, axis=0)
    dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
    active_cluster = int(np.argmin(dists))

    weights = fit_cluster_adaptive_weights(valid_train_items, cluster_of_item,
                                           channel_a_map, channel_b_map, candidate_item_ids,
                                           lr=lr, seed=seed)
    w = weights.get(active_cluster, np.array([0.5, 0.5]))

    active_members = [i for i in valid_train_items if cluster_of_item[i] == active_cluster
                      and i in channel_a_map and i in channel_b_map]
    if not active_members:
        return [], active_cluster, weights
    active_members = active_members[-RECENCY_N:]
    profile_a = normalized(recency_weighted_vec([channel_a_map[i] for i in active_members]))
    profile_b = normalized(recency_weighted_vec([channel_b_map[i] for i in active_members]))

    scores = w[0] * (candidate_a_matrix @ profile_a) + w[1] * (candidate_b_matrix @ profile_b)
    scores = np.asarray(scores).copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:M]
    recs = [candidate_item_ids[i] for i in top_idx]
    return recs, active_cluster, weights
