"""
experiments/movielens/generate_embeddings_movielens.py
------------------------------------------------------------
Step 11c: item embeddings via all-MiniLM-L6-v2 (same model as
Last.fm/Amazon), for the 7,654 items with fetched TMDb text.

Output:
  data/movielens_processed/embeddings_metadata.pkl   ({item_id: np.ndarray(384,)})

Run:
    python experiments/movielens/generate_embeddings_movielens.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "movielens_processed"


def main():
    item_text = pd.read_csv(PROC_DIR / "item_text.csv")
    print(f"[embeddings] Encoding {len(item_text):,} items with all-MiniLM-L6-v2 ...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(item_text["item_text"].tolist(), show_progress_bar=True, batch_size=64)

    emb_map = {int(iid): emb for iid, emb in zip(item_text["movieId"], embeddings)}
    out_path = PROC_DIR / "embeddings_metadata.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(emb_map, f)
    print(f"[embeddings] Saved {len(emb_map):,} embeddings to {out_path}")


if __name__ == "__main__":
    main()
