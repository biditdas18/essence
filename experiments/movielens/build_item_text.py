"""
experiments/movielens/build_item_text.py
--------------------------------------------
Step 11a.6: build item text as "<title> - <genre>, <plot summary>",
matching the Last.fm/Amazon template pattern, from the fetched TMDb
overviews. Also Step 11b: sanity checkpoint before proceeding to
embeddings/training.

Outputs:
  data/movielens_processed/item_text.csv   (movieId, item_text, has_overview)

Run:
    python experiments/movielens/build_item_text.py
"""

import json
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "movielens_processed"

SEED = 42


def clean_genres(genres: str) -> str:
    if pd.isna(genres) or genres == "(no genres listed)":
        return "Unknown"
    return genres.replace("|", ", ")


def main():
    records = []
    with open(PROC_DIR / "tmdb_overviews.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"[build_item_text] Loaded {len(df):,} fetched TMDb records")

    # Original MovieLens title has "(Year)" suffix; strip for readability, keep genres as-is
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    df["clean_genres"] = df["genres"].apply(clean_genres)
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df["has_overview"] = df["overview"].str.len() > 0

    def make_text(row):
        base = f"{row['clean_title']} — {row['clean_genres']}"
        if row["has_overview"]:
            return f"{base}, {row['overview']}"
        return base  # fall back to title+genre only if TMDb had no overview text

    df["item_text"] = df.apply(make_text, axis=1)

    out = df[["movieId", "item_text", "has_overview"]]
    out_path = PROC_DIR / "item_text.csv"
    out.to_csv(out_path, index=False)
    print(f"[build_item_text] Saved {len(out):,} rows to {out_path}")
    print(f"[build_item_text] {df['has_overview'].sum():,}/{len(df):,} "
          f"({100*df['has_overview'].mean():.1f}%) have a non-empty TMDb overview "
          f"(rest fall back to title+genre only)")

    # ── Step 11b sanity checkpoint ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 11b SANITY CHECKPOINT")
    print(f"{'='*70}")

    candidates = pd.read_csv(PROC_DIR / "item_candidates.csv")
    n_total_candidates = len(candidates)
    n_fetched = len(df)
    n_failed = n_total_candidates - n_fetched  # includes missing tmdbId (10) + fetch failures (60)
    success_rate = 100 * n_fetched / n_total_candidates

    print(f"Total candidate items: {n_total_candidates:,}")
    print(f"Successfully fetched:  {n_fetched:,}")
    print(f"Failed/missing:        {n_failed:,}")
    print(f"Success rate:          {success_rate:.1f}%")

    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(df)), 10)
    print(f"\n10 random constructed item texts:")
    empty_or_bad = 0
    for i in sample_idx:
        row = df.iloc[i]
        text = row["item_text"]
        flag = ""
        if len(text.strip()) == 0:
            flag = " [EMPTY]"
            empty_or_bad += 1
        elif "error" in text.lower() or "not found" in text.lower():
            flag = " [SUSPICIOUS - looks like an error message]"
            empty_or_bad += 1
        print(f"  - {text[:200]}{'...' if len(text) > 200 else ''}{flag}")

    print(f"\nBad/empty examples in sample: {empty_or_bad}/10")

    if success_rate < 90:
        print("\n*** STOPPING: success rate below 90% threshold. Do not proceed to embeddings/training. ***")
        raise SystemExit(1)
    else:
        print(f"\nSuccess rate {success_rate:.1f}% >= 90% threshold. Sanity check PASSED. Safe to proceed.")


if __name__ == "__main__":
    main()
