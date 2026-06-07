"""
Step 2 — Generate embeddings for Amazon Books experiment.

PASS 1 — Metadata only (baseline replication):
  input = "title by author. description"  for every unique item
  output: data/amazon_processed/embeddings_metadata.pkl
          dict {item_id: np.ndarray shape (384,)}

PASS 2 — User-review enriched:
  input = review_text[:500]  if user wrote a review for that item
          "title by author. description"  otherwise
  output: data/amazon_processed/embeddings_user_review.pkl
          dict {(user_id, item_id): np.ndarray shape (384,)}
"""

import csv
import pickle
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon_processed"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_train_test():
    rows = []
    for fname in ("train.csv", "test.csv"):
        with open(PROC_DIR / fname) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def load_item_meta():
    meta = {}
    with open(PROC_DIR / "item_meta.csv") as f:
        for row in csv.DictReader(f):
            meta[row["item_id"]] = row
    return meta


def meta_text(item_id: str, meta: dict) -> str:
    m = meta.get(item_id, {})
    return f"{m.get('title','')} by {m.get('author','')}. {m.get('description','')}"


def embed_in_batches(model, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs)
        if (i // batch_size) % 50 == 0 and i > 0:
            print(f"    ... {i:,} / {len(texts):,} encoded")
    return np.vstack(all_vecs)


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading train/test rows and item metadata ...")
    all_rows = load_train_test()
    meta     = load_item_meta()
    print(f"  {len(all_rows):,} total (train+test) rows")
    print(f"  {len(meta):,} items with metadata")

    # ── PASS 1: Metadata embeddings ─────────────────────────────────────────
    print("\n" + "="*55)
    print("PASS 1 — Metadata-only embeddings")
    print("="*55)

    unique_items = sorted({r["item_id"] for r in all_rows})
    print(f"  Unique items to embed: {len(unique_items):,}")

    texts_p1 = [meta_text(iid, meta) for iid in unique_items]

    t0 = time.time()
    vecs_p1 = embed_in_batches(model, texts_p1)
    elapsed_p1 = time.time() - t0

    embeddings_metadata = {iid: vecs_p1[i] for i, iid in enumerate(unique_items)}

    out1 = PROC_DIR / "embeddings_metadata.pkl"
    with open(out1, "wb") as f:
        pickle.dump(embeddings_metadata, f)

    print(f"\n  PASS 1 complete:")
    print(f"    Items embedded  : {len(embeddings_metadata):,}")
    print(f"    Time taken      : {elapsed_p1:.1f}s ({elapsed_p1/60:.1f} min)")
    print(f"    Saved to        : {out1}")

    # ── PASS 2: User-review enriched embeddings ─────────────────────────────
    print("\n" + "="*55)
    print("PASS 2 — User-review enriched embeddings")
    print("="*55)

    pairs      = [(r["user_id"], r["item_id"]) for r in all_rows]
    sources    = [r["embedding_source"] for r in all_rows]
    emb_inputs = [r["embedding_input"] for r in all_rows]

    print(f"  (user, item) pairs to embed: {len(pairs):,}")
    review_count  = sum(1 for s in sources if s == "review")
    meta_count    = len(sources) - review_count
    print(f"  Review-sourced  : {review_count:,}  ({100*review_count/len(sources):.1f}%)")
    print(f"  Metadata-sourced: {meta_count:,}  ({100*meta_count/len(sources):.1f}%)")

    t0 = time.time()
    vecs_p2 = embed_in_batches(model, emb_inputs)
    elapsed_p2 = time.time() - t0

    embeddings_user_review = {pair: vecs_p2[i] for i, pair in enumerate(pairs)}

    out2 = PROC_DIR / "embeddings_user_review.pkl"
    with open(out2, "wb") as f:
        pickle.dump(embeddings_user_review, f)

    print(f"\n  PASS 2 complete:")
    print(f"    (user, item) pairs embedded: {len(embeddings_user_review):,}")
    print(f"    Review-sourced : {review_count:,}  ({100*review_count/len(pairs):.1f}%)")
    print(f"    Metadata-sourced: {meta_count:,}  ({100*meta_count/len(pairs):.1f}%)")
    print(f"    Time taken     : {elapsed_p2:.1f}s ({elapsed_p2/60:.1f} min)")
    print(f"    Saved to       : {out2}")

    # ── CHECKPOINT 2 ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("CHECKPOINT 2 — Embeddings summary")
    print("="*55)
    print(f"  Pass 1 — metadata only:")
    print(f"    Total items embedded : {len(embeddings_metadata):,}")
    print(f"    Time taken           : {elapsed_p1:.1f}s ({elapsed_p1/60:.1f} min)")
    print(f"  Pass 2 — user-review enriched:")
    print(f"    Total (user, item) pairs : {len(embeddings_user_review):,}")
    print(f"    Review source    : {review_count:,}  ({100*review_count/len(pairs):.1f}%)")
    print(f"    Metadata source  : {meta_count:,}  ({100*meta_count/len(pairs):.1f}%)")
    print(f"    Time taken       : {elapsed_p2:.1f}s ({elapsed_p2/60:.1f} min)")
    print(f"  Total time (both passes): {(elapsed_p1+elapsed_p2):.1f}s ({(elapsed_p1+elapsed_p2)/60:.1f} min)")
    print()
    print("CHECKPOINT 2 COMPLETE — waiting for confirmation before Step 3")
    print("="*55)


if __name__ == "__main__":
    main()
