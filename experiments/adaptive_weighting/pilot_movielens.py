"""
experiments/adaptive_weighting/pilot_movielens.py
------------------------------------------------------
Step 2a: MovieLens adaptive-weighting pilot. Same AdaptiveWeightedRecommend
implementation as the Amazon pilot -- channel A = genre-only embedding,
channel B = plot-summary-only embedding (both pure item metadata, no
per-user data, no leakage-equivalent risk -- verified directly against
build_movielens_channels.py before running, not after).

Same 200-user scale, same 4-system comparison, same learning rate (0.05),
same proxy-feedback design as tonight's Amazon pilot. No mechanism changes.

Run:
    python experiments/adaptive_weighting/pilot_movielens.py --n-users 200
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from models.recommenders import build_itemknn_model, cf_itemknn_recommend
from adaptive_weighted_recommend import adaptive_weighted_recommend, normalized, recency_weighted_vec

PROC_DIR = BASE_DIR / "data" / "movielens_processed"
OUT_DIR = Path(__file__).parent

N_PILOT_USERS = 200
M = 10
K = 3
SEED = 42


def recall_at_k(recs, actual, k=10):
    if not actual:
        return 0.0
    return len(set(recs[:k]) & set(actual)) / len(actual)


def lt_recall_at_k(recs, actual, lt_set, k=10):
    actual_lt = [i for i in actual if i in lt_set]
    if not actual_lt:
        return None
    return len(set(recs[:k]) & set(actual_lt)) / len(actual_lt)


def top_k_unseen(scores, seen_mask, item_ids, k=10):
    scores = scores.copy()
    scores[seen_mask] = -2.0
    top_idx = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [item_ids[i] for i in top_idx]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=N_PILOT_USERS)
    parser.add_argument("--disable-update", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    lt_df = pd.read_csv(PROC_DIR / "longtail_items.csv")
    lt_set = set(lt_df["item_id"])

    with open(OUT_DIR / "movielens_channel_a.pkl", "rb") as f:
        channel_a = pickle.load(f)
    with open(OUT_DIR / "movielens_channel_b.pkl", "rb") as f:
        channel_b = pickle.load(f)
    with open(PROC_DIR / "embeddings_metadata.pkl", "rb") as f:
        emb_meta = pickle.load(f)

    item_ids = sorted(set(channel_a.keys()) & set(channel_b.keys()))
    n_items = len(item_ids)
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    print(f"[pilot_movielens] candidate items: {n_items} (channel_a={len(channel_a)}, channel_b={len(channel_b)})")

    A_matrix = normalized(np.array([channel_a[i] for i in item_ids], dtype=np.float32))
    B_matrix = normalized(np.array([channel_b[i] for i in item_ids], dtype=np.float32))
    C_meta = normalized(np.array([emb_meta[i] for i in item_ids if i in emb_meta], dtype=np.float32))
    meta_item_ids = [i for i in item_ids if i in emb_meta]
    meta_item_index = {iid: idx for idx, iid in enumerate(meta_item_ids)}

    rng = np.random.default_rng(SEED)
    all_users = sorted(train_df["user_id"].unique())
    pilot_users = rng.choice(all_users, size=min(args.n_users, len(all_users)), replace=False)

    itemknn = build_itemknn_model(train_df, item_col="item_id")

    test_map = {}
    for uid in pilot_users:
        test_map[uid] = test_df[test_df["user_id"] == uid]["item_id"].tolist()

    systems = ["Essence (K=3)", "Recency-Weighted", "CF (ItemKNN)", "AdaptiveWeighted"]
    recall = {s: [] for s in systems}
    lt_recall = {s: [] for s in systems}
    learned_weights = []

    for uid in tqdm(pilot_users, desc="Pilot users"):
        user_rows = train_df[train_df["user_id"] == uid].sort_values("timestamp")
        train_items = list(user_rows["item_id"])
        seen_set = set(train_items)
        actual = test_map[uid]

        seen_mask_meta = np.zeros(len(meta_item_ids), dtype=bool)
        for iid in seen_set:
            idx = meta_item_index.get(iid)
            if idx is not None:
                seen_mask_meta[idx] = True

        seen_mask_channels = np.zeros(n_items, dtype=bool)
        for iid in seen_set:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask_channels[idx] = True

        recs_cf = cf_itemknn_recommend(uid, train_df, itemknn, M)

        seen_vecs = [emb_meta[i] for i in train_items if i in emb_meta]
        if len(seen_vecs) >= K:
            km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
            km.fit(np.array(seen_vecs))
            recent_vecs = seen_vecs[-10:]
            recent_mean = np.mean(recent_vecs, axis=0)
            dists = np.linalg.norm(km.cluster_centers_ - recent_mean, axis=1)
            centroid = normalized(km.cluster_centers_[np.argmin(dists)].astype(np.float32))
            scores_essence = C_meta @ centroid
            recs_essence = top_k_unseen(scores_essence, seen_mask_meta, meta_item_ids, M)
        else:
            recs_essence = []

        recent_vecs = [emb_meta[i] for i in train_items[-10:] if i in emb_meta]
        if recent_vecs:
            rw_vec = normalized(recency_weighted_vec(recent_vecs))
            scores_rw = C_meta @ rw_vec
            recs_rw = top_k_unseen(scores_rw, seen_mask_meta, meta_item_ids, M)
        else:
            recs_rw = []

        recs_adaptive, w = adaptive_weighted_recommend(
            uid, train_items, channel_a, channel_b, item_ids, A_matrix, B_matrix,
            seen_mask_channels, M=M, seed=SEED, disable_update=args.disable_update,
        )
        learned_weights.append(w)

        for name, recs in zip(systems, [recs_essence, recs_rw, recs_cf, recs_adaptive]):
            recall[name].append(recall_at_k(recs, actual, k=M))
            ltr = lt_recall_at_k(recs, actual, lt_set, k=M)
            if ltr is not None:
                lt_recall[name].append(ltr)

    mode_label = "SANITY CHECK: weights fixed at (0.5, 0.5)" if args.disable_update else "learned weights"
    print(f"\n{'='*70}")
    print(f"PILOT RESULT -- MovieLens, {len(pilot_users)} users, raw numbers (NO bootstrap) -- {mode_label}")
    print(f"{'='*70}")
    print(f"{'System':<20} {'Recall@10':>10} {'LT-Recall@10':>13}")
    for name in systems:
        r = np.mean(recall[name])
        lt = np.mean(lt_recall[name]) if lt_recall[name] else float("nan")
        print(f"{name:<20} {r:>10.4f} {lt:>13.4f}")

    w_arr = np.array(learned_weights)
    print(f"\nLearned weights (w_A=genre, w_B=plot) across {len(w_arr)} users:")
    print(f"  mean: w_A={w_arr[:,0].mean():.3f}  w_B={w_arr[:,1].mean():.3f}")
    print(f"  std:  w_A={w_arr[:,0].std():.3f}  w_B={w_arr[:,1].std():.3f}")

    rw_recall = np.mean(recall["Recency-Weighted"])
    adaptive_recall = np.mean(recall["AdaptiveWeighted"])
    rw_lt = np.mean(lt_recall["Recency-Weighted"]) if lt_recall["Recency-Weighted"] else float("nan")
    adaptive_lt = np.mean(lt_recall["AdaptiveWeighted"]) if lt_recall["AdaptiveWeighted"] else float("nan")

    print(f"\n{'='*70}")
    print("GO/NO-GO CHECK vs Recency-Weighted (the bar to clear)")
    print(f"{'='*70}")
    print(f"  Recall@10:    AdaptiveWeighted {adaptive_recall:.4f} vs Recency-Weighted {rw_recall:.4f} "
          f"-> {'DIRECTIONAL WIN' if adaptive_recall > rw_recall else 'NO WIN'}")
    print(f"  LT-Recall@10: AdaptiveWeighted {adaptive_lt:.4f} vs Recency-Weighted {rw_lt:.4f} "
          f"-> {'DIRECTIONAL WIN' if adaptive_lt > rw_lt else 'NO WIN'}")


if __name__ == "__main__":
    main()
