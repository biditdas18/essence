"""
evaluation/compute_cost.py
------------------------------
Step 10: wall-clock disclosure table for the limitations section, not a
systems optimization task. Reports per-user latency for:
  - Essence's K-means clustering step in isolation
  - each method's full inference pass (produce top-10 recs for one user)
on both dataset scales (Last.fm: 99 users / 22,767 items; Amazon: 2,000
users / 61,727 items).

MIND/ComiRec timing uses freshly-instantiated (untrained) models -- forward-
pass compute cost is identical whether weights are trained or not, so this
is a valid latency measurement without needing a saved checkpoint. Training
time for MIND/ComiRec is already reported separately in
results/mind_comirec_summary_*.csv (train_seconds column).

Run:
    python evaluation/compute_cost.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import (
    random_recommend, popularity_recommend, build_itemknn_model, cf_itemknn_recommend,
    content_recommend, essence_recommend, last_item_recommend, avg_last10_recommend,
    recency_weighted_recommend,
)
sys.path.insert(0, str(BASE_DIR / "experiments" / "mind_comirec"))
from model import MIND, ComiRecSA

N_SAMPLE_USERS = 20
RESULTS_DIR = BASE_DIR / "results"


def time_calls(fn, n=N_SAMPLE_USERS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def bench_lastfm():
    train_df = pd.read_pickle(BASE_DIR / "data" / "train_interactions.pkl")
    with open(BASE_DIR / "embeddings" / "item_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    users = sorted(train_df["user_id"].unique())[:N_SAMPLE_USERS]
    itemknn = build_itemknn_model(train_df, item_col="track_id")

    rows = []

    # Essence KMeans-only timing (isolated fit call)
    def kmeans_only():
        uid = np.random.choice(users)
        seen = train_df[train_df["user_id"] == uid].sort_values("timestamp")["track_id"]
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) >= 3:
            KMeans(n_clusters=3, random_state=42, n_init=10).fit(np.array(vecs))
    mean_t, std_t = time_calls(kmeans_only)
    rows.append({"dataset": "lastfm", "component": "Essence: K-means fit only", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    systems = {
        "Random": lambda uid: random_recommend(uid, train_df, 10, item_embedding_map=emb),
        "Popularity": lambda uid: popularity_recommend(uid, train_df, 10, item_embedding_map=emb),
        "CF (ItemKNN)": lambda uid: cf_itemknn_recommend(uid, train_df, itemknn, 10, item_embedding_map=emb),
        "Content": lambda uid: content_recommend(uid, train_df, emb, 10),
        "Essence (full)": lambda uid: essence_recommend(uid, train_df, emb, K=3, M=10),
        "Last-Item": lambda uid: last_item_recommend(uid, train_df, emb, 10),
        "Avg-Last-10": lambda uid: avg_last10_recommend(uid, train_df, emb, 10),
        "Recency-Weighted": lambda uid: recency_weighted_recommend(uid, train_df, emb, 10),
    }
    for name, fn in systems.items():
        u_iter = iter(users * 5)
        mean_t, std_t = time_calls(lambda: fn(next(u_iter)))
        rows.append({"dataset": "lastfm", "component": f"{name}: full inference/user", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    # MIND / ComiRec forward-pass timing (untrained weights -- compute cost is weight-independent)
    item_ids = sorted(emb.keys())
    item2idx = {iid: i + 1 for i, iid in enumerate(item_ids)}
    emb_matrix = np.array([emb[i] for i in item_ids], dtype=np.float32)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    for model_name, ModelCls, kwargs in [
        ("MIND", MIND, {"routing_iters": 3}),
        ("ComiRec (SA)", ComiRecSA, {}),
    ]:
        model = ModelCls(n_items=len(item_ids), embedding_dim=384, K=4,
                         item_embedding_init=emb_matrix, max_seq_len=50, **kwargs).to(device)
        model.eval()
        C = torch.from_numpy(emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)).float().to(device)

        def infer():
            uid = np.random.choice(users)
            seen = list(train_df[train_df["user_id"] == uid].sort_values("timestamp")["track_id"])
            idxs = [item2idx[i] for i in seen if i in item2idx][-50:]
            padded = idxs + [0] * (50 - len(idxs))
            with torch.no_grad():
                seq_t = torch.tensor([padded], dtype=torch.long, device=device)
                interests = model.extract_interests(seq_t)[0]
                interests_n = interests / (interests.norm(dim=-1, keepdim=True) + 1e-8)
                _ = (C @ interests_n.T).max(dim=1)
        mean_t, std_t = time_calls(infer)
        rows.append({"dataset": "lastfm", "component": f"{model_name}: full inference/user (untrained weights)", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    return rows


def bench_amazon():
    proc_dir = BASE_DIR / "data" / "amazon_processed"
    train_df = pd.read_csv(proc_dir / "train.csv")
    if "timestamp" not in train_df.columns:
        train_df["timestamp"] = train_df.index
    with open(proc_dir / "embeddings_metadata.pkl", "rb") as f:
        emb = pickle.load(f)
    users = sorted(train_df["user_id"].unique())[:N_SAMPLE_USERS]

    item_ids = sorted(emb.keys())
    item_index = {iid: i for i, iid in enumerate(item_ids)}
    C_np = np.array([emb[i] for i in item_ids], dtype=np.float32)
    C_np /= (np.linalg.norm(C_np, axis=1, keepdims=True) + 1e-8)

    def top_k_unseen(q, seen_mask, k=10):
        scores = C_np @ q
        scores = scores.copy()
        scores[seen_mask] = -2.0
        return np.lexsort((np.arange(len(scores)), -scores))[:k]

    rows = []

    def kmeans_only():
        uid = np.random.choice(users)
        seen = train_df[train_df["user_id"] == uid].sort_values("timestamp")["item_id"]
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) >= 3:
            KMeans(n_clusters=3, random_state=42, n_init=10).fit(np.array(vecs))
    mean_t, std_t = time_calls(kmeans_only)
    rows.append({"dataset": "amazon", "component": "Essence: K-means fit only", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    def essence_full():
        uid = np.random.choice(users)
        seen = list(train_df[train_df["user_id"] == uid].sort_values("timestamp")["item_id"])
        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        vecs = [emb[i] for i in seen if i in emb]
        if len(vecs) >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(np.array(vecs))
            centroid = km.cluster_centers_[0].astype(np.float32)
            centroid /= (np.linalg.norm(centroid) + 1e-8)
            top_k_unseen(centroid, seen_mask)
    mean_t, std_t = time_calls(essence_full)
    rows.append({"dataset": "amazon", "component": "Essence (full): full inference/user", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    def content_full():
        uid = np.random.choice(users)
        seen = list(train_df[train_df["user_id"] == uid]["item_id"])
        seen_mask = np.zeros(len(item_ids), dtype=bool)
        for iid in seen:
            idx = item_index.get(iid)
            if idx is not None:
                seen_mask[idx] = True
        vecs = [emb[i] for i in seen if i in emb]
        if vecs:
            v = np.mean(vecs, axis=0).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-8)
            top_k_unseen(v, seen_mask)
    mean_t, std_t = time_calls(content_full)
    rows.append({"dataset": "amazon", "component": "Content: full inference/user", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    C_t = torch.from_numpy(C_np).float().to(device)
    item2idx = {iid: i + 1 for i, iid in enumerate(item_ids)}
    emb_matrix = C_np  # already built/normalized above; fine as init for latency purposes

    for model_name, ModelCls, kwargs in [
        ("MIND", MIND, {"routing_iters": 3}),
        ("ComiRec (SA)", ComiRecSA, {}),
    ]:
        model = ModelCls(n_items=len(item_ids), embedding_dim=384, K=4,
                         item_embedding_init=emb_matrix, max_seq_len=50, **kwargs).to(device)
        model.eval()

        def infer():
            uid = np.random.choice(users)
            seen = list(train_df[train_df["user_id"] == uid].sort_values("timestamp")["item_id"])
            idxs = [item2idx[i] for i in seen if i in item2idx][-50:]
            padded = idxs + [0] * (50 - len(idxs))
            with torch.no_grad():
                seq_t = torch.tensor([padded], dtype=torch.long, device=device)
                interests = model.extract_interests(seq_t)[0]
                interests_n = interests / (interests.norm(dim=-1, keepdim=True) + 1e-8)
                _ = (C_t @ interests_n.T).max(dim=1)
        mean_t, std_t = time_calls(infer)
        rows.append({"dataset": "amazon", "component": f"{model_name}: full inference/user (untrained weights)", "mean_ms": mean_t * 1000, "std_ms": std_t * 1000})

    return rows


def main():
    print("[compute_cost] Benchmarking Last.fm (99 users, 22,767 items)...")
    rows = bench_lastfm()
    print("[compute_cost] Benchmarking Amazon (2,000 users, 61,727 items)...")
    rows += bench_amazon()

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    out_path = RESULTS_DIR / "compute_cost.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
