"""
Embed the Last.fm tracks needed to lift the MAX_INTERACTIONS=500 user cap.

WHY THIS EXISTS
---------------
The published Last.fm-1K cohort is 99 users, which is the single largest limit on
statistical power in the paper. That is NOT a property of the dataset: 975 of 992
users clear the >=50 floor, and the 500-track ceiling discards about 90% of them.
Lifting it needs embeddings for tracks outside the published catalogue.

TRANCHED, RESUMABLE DESIGN
--------------------------
Embedding every track needed for the uncapped cohort is ~940k items (~8h on CPU).
Rather than an all-or-nothing job, tracks are ordered by the SMALLEST cap at which
they are required, so the work completes cohort-by-cohort: once the cap=1000
tranche is done, a complete 193-user cohort is available even if the job is stopped.
Progress is flushed to disk after every batch and re-running skips what exists,
so an interrupted run resumes.

Writes embeddings/item_embeddings_expanded.pkl (a superset of item_embeddings.pkl)
and embeddings/expanded_coverage.json recording which caps are fully covered.
"""
from pathlib import Path
import json
import pickle
import sys
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "lastfm-dataset-1K" / \
    "userid-timestamp-artid-artname-traid-traname.tsv"
PUBLISHED = BASE / "embeddings" / "item_embeddings.pkl"
OUT = BASE / "embeddings" / "item_embeddings_expanded.pkl"
COVERAGE = BASE / "embeddings" / "expanded_coverage.json"

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_INTERACTIONS = 50
CAPS = [1000, 1500, 2000, 3000, 5000, None]   # None = no upper bound
BATCH = 4096


def cohort_tracks():
    """Replicates data/preprocess.py's clean+filter, then returns, for each cap,
    the set of track_ids that cohort needs. Order matters: identical logic to the
    published pipeline, or the expanded cohort would not be comparable."""
    df = pd.read_csv(RAW, sep="\t", encoding="utf-8", on_bad_lines="skip",
                     names=["user_id", "timestamp", "artist_id",
                            "artist_name", "track_id", "track_name"])
    df = df.dropna(subset=["track_id", "track_name"])
    df = df.drop_duplicates(subset=["user_id", "track_id"], keep="first")
    df["item_text"] = df["track_name"] + " by " + df["artist_name"].fillna("Unknown")
    counts = df.groupby("user_id")["track_id"].count()

    text_of = (df.drop_duplicates(subset=["track_id"])
                 .set_index("track_id")["item_text"].to_dict())
    per_cap = {}
    for cap in CAPS:
        hi = counts.max() if cap is None else cap
        users = counts[(counts >= MIN_INTERACTIONS) & (counts <= hi)].index
        per_cap[cap] = set(df.loc[df.user_id.isin(users), "track_id"].unique())
    return per_cap, text_of


def main():
    per_cap, text_of = cohort_tracks()

    with open(PUBLISHED, "rb") as fh:
        emb = pickle.load(fh)
    n_published = len(emb)
    if OUT.exists():                      # resume
        with open(OUT, "rb") as fh:
            emb.update(pickle.load(fh))
    print(f"[expand] published embeddings: {n_published:,} | on disk now: {len(emb):,}")

    # order tracks by the smallest cap that requires them
    ordered, seen = [], set(emb)
    for cap in CAPS:
        need = [t for t in per_cap[cap] if t not in seen]
        seen.update(need)
        ordered.extend(need)
        print(f"[expand] cap={cap or 'none':>5}: cohort needs {len(per_cap[cap]):>7,} tracks, "
              f"{len(need):>7,} new in this tranche")
    if not ordered:
        print("[expand] nothing to do; all cohorts already covered")
    else:
        print(f"[expand] total to embed: {len(ordered):,}")
        model = SentenceTransformer(MODEL_NAME)
        t0 = time.time()
        for i in range(0, len(ordered), BATCH):
            chunk = ordered[i:i + BATCH]
            vecs = model.encode([text_of[t] for t in chunk],
                                batch_size=256, show_progress_bar=False,
                                convert_to_numpy=True)
            emb.update(dict(zip(chunk, vecs)))
            with open(OUT, "wb") as fh:          # flush every batch so a kill is safe
                pickle.dump(emb, fh)
            done = min(i + BATCH, len(ordered))
            rate = done / max(1e-9, time.time() - t0)
            print(f"[expand] {done:,}/{len(ordered):,}  ({rate:.0f}/s, "
                  f"eta {(len(ordered)-done)/max(rate,1e-9)/60:.0f} min)", flush=True)

    covered = {str(cap or "none"): bool(per_cap[cap] <= set(emb)) for cap in CAPS}
    covered["500_published"] = True
    with open(COVERAGE, "w") as fh:
        json.dump({"n_embeddings": len(emb), "caps_fully_covered": covered}, fh, indent=1)
    print("[expand] caps fully covered:", {k: v for k, v in covered.items() if v})


if __name__ == "__main__":
    sys.exit(main())
