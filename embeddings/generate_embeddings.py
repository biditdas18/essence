"""
embeddings/generate_embeddings.py
----------------------------------
Generates all-MiniLM-L6-v2 embeddings for every unique track
across BOTH the training and test sets.

Why embed test items too?
-------------------------
The Last.fm-1K dataset is extremely sparse after filtering (89% of tracks
are singletons — heard by exactly one user).  With a random hold-out split,
a user's test tracks are often unique to them, so they never appear in any
other user's train interactions.  If we embed only train items, ~80% of test
items fall outside the embedding map and can never be recovered by any
recommender, making recall trivially near-zero.

By embedding the full item universe (train ∪ test) we ensure every candidate
item can be scored.  This is legitimate: embeddings use only track metadata
(name + artist), not any interaction signal — no label leakage occurs.
The recommenders still exclude the user's own train items from the candidate
pool; the test items remain "unseen" and are valid recommendation targets.

Output
------
  embeddings/item_embeddings.pkl   — dict {track_id: np.ndarray (384,)}

Run:
    python embeddings/generate_embeddings.py
"""

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent.parent
TRAIN_PATH      = BASE_DIR / "data" / "train_interactions.pkl"
TEST_PATH       = BASE_DIR / "data" / "test_interactions.pkl"
EMBEDDINGS_OUT  = Path(__file__).parent / "item_embeddings.pkl"

BATCH_SIZE      = 256
MODEL_NAME      = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_PATH}. "
            "Run: python data/preprocess.py"
        )

    import pandas as pd
    train_df = pd.read_pickle(TRAIN_PATH)
    test_df  = pd.read_pickle(TEST_PATH) if TEST_PATH.exists() else pd.DataFrame()

    # Combine train + test to get the full item universe
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Unique (track_id, item_text) pairs across both splits
    unique_items = (
        combined[["track_id", "item_text"]]
        .drop_duplicates(subset=["track_id"])
        .reset_index(drop=True)
    )
    print(f"[embed] Train unique tracks : {train_df['track_id'].nunique():,}")
    print(f"[embed] Test  unique tracks : {test_df['track_id'].nunique() if len(test_df) else 0:,}")
    print(f"[embed] Combined unique     : {len(unique_items):,}  (train ∪ test)")
    print(f"[embed] Unique tracks to embed: {len(unique_items):,}")

    # 3. Load model
    print(f"[embed] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Encode in batches of 256
    texts = unique_items["item_text"].tolist()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # 5. Build {track_id: embedding} dict and save
    embedding_map = {
        tid: vec
        for tid, vec in zip(unique_items["track_id"], vectors)
    }

    with open(EMBEDDINGS_OUT, "wb") as fh:
        pickle.dump(embedding_map, fh)

    # 6. Print summary
    sample_vec = next(iter(embedding_map.values()))
    print(f"\n[embed] Total items embedded  : {len(embedding_map):,}")
    print(f"[embed] Embedding dimension   : {sample_vec.shape[0]}")
    print(f"[embed] Saved to              : {EMBEDDINGS_OUT}")


if __name__ == "__main__":
    main()
