"""
experiments/adaptive_weighting/build_movielens_channels.py
------------------------------------------------------------------
Step 1: build 2 separate per-item embedding channels for MovieLens:
  channel A = genre-only text ("Comedy, Drama")
  channel B = plot-summary-only text (TMDb overview)
Both embedded with the same all-MiniLM-L6-v2 model already used
throughout this repo. No title/creator text in either channel -- this
isolates "what genre" from "what happens" as the two signal types.

Run:
    python experiments/adaptive_weighting/build_movielens_channels.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "movielens_processed"
OUT_DIR = Path(__file__).parent


def clean_genres(genres):
    if pd.isna(genres) or genres == "(no genres listed)":
        return "Unknown"
    return genres.replace("|", ", ")


def main():
    records = []
    with open(PROC_DIR / "tmdb_overviews.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["genre_text"] = df["genres"].apply(clean_genres)
    df["plot_text"] = df["overview"].fillna("").astype(str).str.strip()

    print(f"[build_channels] Encoding {len(df)} genre-only texts (channel A) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    genre_embs = model.encode(df["genre_text"].tolist(), show_progress_bar=True, batch_size=64)
    channel_a = {int(mid): emb for mid, emb in zip(df["movieId"], genre_embs)}

    print(f"[build_channels] Encoding {len(df)} plot-only texts (channel B) ...")
    plot_embs = model.encode(df["plot_text"].tolist(), show_progress_bar=True, batch_size=64)
    channel_b = {int(mid): emb for mid, emb in zip(df["movieId"], plot_embs)}

    with open(OUT_DIR / "movielens_channel_a.pkl", "wb") as f:
        pickle.dump(channel_a, f)
    with open(OUT_DIR / "movielens_channel_b.pkl", "wb") as f:
        pickle.dump(channel_b, f)
    print(f"[build_channels] Saved {len(channel_a)} channel-A and {len(channel_b)} channel-B embeddings")

    import numpy as np
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(sorted(channel_a.keys()), 5, replace=False)
    print("\nMovieLens spot-check (channel A vs channel B cosine similarity):")
    for mid in sample_ids:
        a, b = channel_a[mid], channel_b[mid]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        row = df[df["movieId"] == mid].iloc[0]
        print(f"  movieId {mid} ({row['title']}): genre='{row['genre_text']}' cos_sim(A,B)={cos:.4f}")


if __name__ == "__main__":
    main()
