"""
Step 1 — Preprocess Amazon Reviews 2023 Books 5-core dataset.

Inputs  (data/amazon_raw/):
  Books_5core_ratings.csv       — user_id, parent_asin, rating, timestamp
  Books_reviews_filtered.jsonl  — user_id, item_id, review_text, timestamp
  meta_Books_filtered.jsonl     — item_id, title, author, description

Outputs (data/amazon_processed/):
  train.csv         — user_id, item_id, embedding_input, embedding_source
  test.csv          — user_id, item_id, embedding_input, embedding_source
  item_meta.csv     — item_id, title, author, description
  longtail_items.csv — item_id (singleton train items only)
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

RAW_DIR  = Path(__file__).resolve().parents[2] / "data" / "amazon_raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon_processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_FILE = RAW_DIR / "Books_5core_ratings.csv"
REVIEWS_FILE = RAW_DIR / "Books_reviews_filtered.jsonl"
META_FILE    = RAW_DIR / "meta_Books_filtered.jsonl"

SEED = 42
MIN_INTERACTIONS = 20
MAX_INTERACTIONS = 200
MAX_USERS        = 2000


# ─── 1. Load metadata ────────────────────────────────────────────────────────
print("Loading metadata ...")
meta = {}  # item_id -> {title, author, description}
with open(META_FILE) as f:
    for line in f:
        row = json.loads(line)
        meta[row["item_id"]] = {
            "title":       row.get("title", ""),
            "author":      row.get("author", ""),
            "description": row.get("description", ""),
        }
print(f"  Loaded {len(meta):,} items with metadata")


# ─── 2. Load ratings (5-core) ─────────────────────────────────────────────────
print("Loading ratings ...")
# user -> list of (item_id, timestamp) in file order (dedup later)
user_items_raw = defaultdict(list)
total_ratings = 0
with open(RATINGS_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid  = row["user_id"]
        asin = row["parent_asin"]
        ts   = row["timestamp"]
        user_items_raw[uid].append((asin, ts))
        total_ratings += 1
print(f"  Loaded {total_ratings:,} ratings for {len(user_items_raw):,} users")


# ─── 3. Load reviews → (user_id, item_id) -> review_text ─────────────────────
print("Loading reviews (streaming) ...")
reviews = {}  # (user_id, item_id) -> review_text
with open(REVIEWS_FILE) as f:
    for line in f:
        row = json.loads(line)
        uid = row.get("user_id", "")
        iid = row.get("item_id", "")
        txt = row.get("review_text", "")
        key = (uid, iid)
        # Keep first occurrence (consistent with dedup logic below)
        if key not in reviews:
            reviews[key] = txt
print(f"  Loaded {len(reviews):,} (user, item) review pairs")


# ─── 4. Deduplicate (user_id, item_id) — keep first occurrence ───────────────
print("Deduplicating and filtering users ...")
user_items = {}  # user_id -> list of (item_id, timestamp) unique items
for uid, items in user_items_raw.items():
    seen = set()
    deduped = []
    for (iid, ts) in items:
        if iid not in seen:
            seen.add(iid)
            deduped.append((iid, ts))
    user_items[uid] = deduped

# Filter to users with 20–200 unique items
filtered_users = {
    uid: items
    for uid, items in user_items.items()
    if MIN_INTERACTIONS <= len(items) <= MAX_INTERACTIONS
}
print(f"  Users after 20–200 filter: {len(filtered_users):,}")


# ─── 5. Subsample to 2000 users (seed 42) ─────────────────────────────────────
rng = random.Random(SEED)
if len(filtered_users) > MAX_USERS:
    sampled_uids = rng.sample(sorted(filtered_users.keys()), MAX_USERS)
else:
    sampled_uids = sorted(filtered_users.keys())
sampled_users = {uid: filtered_users[uid] for uid in sampled_uids}
print(f"  Users after subsample: {len(sampled_users):,}")


# ─── 6. Chronological 80/20 split per user ────────────────────────────────────
# Sort each user's items by timestamp ascending; first 80% = train, last 20% = test.
# This is consistent with recommenders.py's active-centroid selection, which uses
# the chronologically LAST r items of the train set (seen[-10:] after sort_values).
print("Splitting train/test (chronological) ...")
train_records = []  # list of (user_id, item_id, timestamp)
test_records  = []

for uid in sampled_uids:
    items = sampled_users[uid]  # list of (item_id, timestamp) — deduped
    # Sort by timestamp ascending; timestamps are ms-epoch integers as strings
    items_sorted = sorted(items, key=lambda x: int(x[1]) if str(x[1]).isdigit() else x[1])
    n_test  = max(1, round(len(items_sorted) * 0.20))
    n_train = len(items_sorted) - n_test
    for iid, ts in items_sorted[:n_train]:
        train_records.append((uid, iid, ts))
    for iid, ts in items_sorted[n_train:]:
        test_records.append((uid, iid, ts))

print(f"  Train: {len(train_records):,}  Test: {len(test_records):,}")


# ─── 7. Long-tail: items with exactly 1 interaction in train ──────────────────
from collections import Counter
train_item_counts = Counter(iid for (_, iid, _) in train_records)
longtail_items = {iid for iid, cnt in train_item_counts.items() if cnt == 1}
print(f"  Singleton (long-tail) train items: {len(longtail_items):,}")


# ─── 8. Build embedding_input and embedding_source ────────────────────────────
print("Building embedding inputs ...")

def build_embedding(uid, iid, ts):
    review_text = reviews.get((uid, iid), "")
    if review_text and review_text.strip():
        return review_text[:500], "review"
    else:
        m = meta.get(iid, {})
        title  = m.get("title", "")
        author = m.get("author", "")
        desc   = m.get("description", "")
        return f"{title} by {author}. {desc}", "metadata"

# Track join stats across all records
meta_join_hits   = 0
review_join_hits = 0
ratings_total    = len(train_records) + len(test_records)

train_rows = []
for uid, iid, ts in train_records:
    emb_input, emb_src = build_embedding(uid, iid, ts)
    if meta.get(iid):
        meta_join_hits += 1
    if reviews.get((uid, iid)):
        review_join_hits += 1
    train_rows.append({
        "user_id": uid, "item_id": iid,
        "embedding_input": emb_input, "embedding_source": emb_src,
    })

test_rows = []
for uid, iid, ts in test_records:
    emb_input, emb_src = build_embedding(uid, iid, ts)
    if meta.get(iid):
        meta_join_hits += 1
    if reviews.get((uid, iid)):
        review_join_hits += 1
    test_rows.append({
        "user_id": uid, "item_id": iid,
        "embedding_input": emb_input, "embedding_source": emb_src,
    })

all_rows = train_rows + test_rows
review_pct   = 100 * sum(1 for r in all_rows if r["embedding_source"] == "review") / len(all_rows)
metadata_pct = 100 - review_pct


# ─── 9. Save outputs ──────────────────────────────────────────────────────────
print("Saving outputs ...")

COLS = ["user_id", "item_id", "embedding_input", "embedding_source"]

with open(PROC_DIR / "train.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(train_rows)

with open(PROC_DIR / "test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(test_rows)

# item_meta: all items that appear in train or test
active_items = {r["item_id"] for r in all_rows}
with open(PROC_DIR / "item_meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["item_id", "title", "author", "description"])
    w.writeheader()
    for iid in sorted(active_items):
        m = meta.get(iid, {})
        w.writerow({
            "item_id":     iid,
            "title":       m.get("title", ""),
            "author":      m.get("author", ""),
            "description": m.get("description", ""),
        })

with open(PROC_DIR / "longtail_items.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["item_id"])
    w.writeheader()
    for iid in sorted(longtail_items):
        w.writerow({"item_id": iid})

print("  All files saved.")


# ─── CHECKPOINT 1 ─────────────────────────────────────────────────────────────
total_users     = len(sampled_users)
total_items     = len(active_items)
avg_per_user    = ratings_total / total_users
catalog_pct_lt  = 100 * len(longtail_items) / total_items
meta_join_rate  = 100 * meta_join_hits   / ratings_total
review_join_rate= 100 * review_join_hits / ratings_total

# Both-join rate: ratings rows that joined to BOTH a review AND metadata
both_hits = sum(
    1 for r in all_rows
    if meta.get(r["item_id"]) and reviews.get((r["user_id"], r["item_id"]))
)
both_join_rate = 100 * both_hits / ratings_total

print()
print("=" * 60)
print("CHECKPOINT 1 — Preprocessing summary")
print("=" * 60)
print(f"  1. Total users (after filter + subsample) : {total_users:,}")
print(f"  2. Total unique items                      : {total_items:,}")
print(f"  3. Avg interactions per user               : {avg_per_user:.1f}")
print(f"  4. % with review coverage (source=review)  : {review_pct:.1f}%")
print(f"  5. % using metadata fallback               : {metadata_pct:.1f}%")
print(f"  6. Singleton items (train)                 : {len(longtail_items):,}  ({catalog_pct_lt:.1f}% of catalog)")
print(f"  7. Train size                              : {len(train_rows):,}")
print(f"     Test size                               : {len(test_rows):,}")
print(f"  8. Metadata join rate                      : {meta_join_rate:.1f}%")
print(f"     Review join rate                        : {review_join_rate:.1f}%")
print(f"     Both-joined rate (review + meta)        : {both_join_rate:.1f}%")
print()
print("CHECKPOINT 1 COMPLETE — waiting for confirmation before Step 2")
print("=" * 60)
