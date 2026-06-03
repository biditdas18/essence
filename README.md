# Essence: Personal Embedding Cluster Framework

> **A recommendation system that knows you in depth — not just in breadth.**

---

## What Is Essence?

Most recommendation systems work by comparing you to other people. They find users who behave like you, then recommend what those users liked. This sounds reasonable — but it has a fundamental flaw: it only knows the parts of you that overlap with the crowd.

**Essence works differently.** It never looks at any other user. Instead, it builds a private semantic map of *your* listening history, clusters that map into distinct interest profiles, and recommends music that matches whichever profile you're currently in.

The result: instead of recommending what's popular among people vaguely like you, Essence recommends what *you* specifically would reach for next — including the obscure, niche, and deeply personal tracks that no crowd would ever surface.

---

## The Problem With Collaborative Filtering

Collaborative Filtering (CF) is the engine behind most major recommendation systems. It works like this:

1. Collect millions of users' listening histories
2. Find users whose history overlaps with yours
3. Recommend what those similar users liked that you haven't heard yet

**The problem:** CF's recommendations are only as good as its ability to find users like you — and it can only find similarity on items many people listen to. Popular items accumulate lots of co-occurrence signal. Niche items have almost none.

**Concrete example:** Imagine a user who listens to both mainstream pop and obscure 1970s Scandinavian progressive rock. CF will find thousands of users who share the pop taste and recommend more pop. It will find almost nobody who shares the Scandinavian prog taste, so it will never recommend anything from that world — even if that's actually the side of the user's taste that's deepest and most intentional.

This is called **popularity bias**: CF systematically over-recommends popular items and under-recommends niche ones.

---

## How Essence Works

Essence replaces cross-user similarity with **within-user clustering**. Here's the full pipeline in plain language:

**Step 1 — Embed every track.**
Each track in the user's history is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`, a pre-trained sentence embedding model. The input is simply `"<track name> by <artist name>"`. Tracks that are semantically similar (same genre, era, mood, style) end up close together in this space.

**Step 2 — Cluster the user's history into K=3 interest profiles.**
K-means clustering groups the user's embedded tracks into 3 clusters. Each cluster centroid is a point in embedding space that represents the "centre of gravity" of one of the user's distinct taste areas. For our example user, one centroid might represent pop, another might represent progressive rock, another might represent something else entirely.

**Step 3 — Identify the active cluster.**
We take the user's 10 most recent listening events and compute their average embedding — a "recency vector" that represents what they've been into lately. We then find whichever of the 3 centroids is closest to this recency vector. That's the **active centroid**: the interest profile the user is currently expressing.

**Step 4 — Recommend by similarity to the active centroid.**
Every track the user hasn't heard is scored by cosine similarity to the active centroid. The top-10 tracks are returned as recommendations. These will be tracks that are semantically close to the user's current taste cluster — not globally popular tracks, and not a blurry average of all their taste, but specifically what matches the depth they're currently in.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   USER'S LISTENING HISTORY               │
│          (tracks sorted chronologically)                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Embedding Model             │
          │  all-MiniLM-L6-v2 (384-dim) │
          │                              │
          │  input:  "Track by Artist"   │
          │  output: unit vector ℝ³⁸⁴   │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Item Embedding Matrix        │
          │  shape: (n_tracks × 384)      │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  K-Means Clustering  (K=3)   │
          └──────┬───────────┬───────────┘
                 │           │           │
                 ▼           ▼           ▼
              ┌─────┐    ┌─────┐    ┌─────┐
              │ C1  │    │ C2  │    │ C3  │
              │     │    │     │    │     │
              │Jazz │    │Pop  │    │Prog │
              │taste│    │taste│    │rock │
              └─────┘    └─────┘    └─────┘
                 Interest Profile Centroids
                         │
          ┌──────────────▼───────────────┐
          │   Active Cluster Selection    │
          │                              │
          │   mean(last 10 tracks)        │
          │        → closest centroid     │
          └──────────────┬───────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  Active Centroid  │   ← "what the user
               │  (e.g., C3: Prog) │      is into RIGHT NOW"
               └────────┬─────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Cosine Similarity Search    │
          │  against all unseen tracks   │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Top-10 Recommendations      │
          │  (niche tracks welcome)       │
          └──────────────────────────────┘
```

---

## How Essence Compares to Prior Work

Other multi-interest recommendation systems exist — but they all require cross-user data:

| System       | Multi-Interest | Cross-User Data | Session-Aware | Privacy-Safe |
|:-------------|:--------------:|:---------------:|:-------------:|:------------:|
| MIND (2019)  | ✅             | ✅ Required     | ❌            | ❌           |
| ComiRec (2020)| ✅            | ✅ Required     | ❌            | ❌           |
| BERT4Rec (2019)| ❌           | ✅ Required     | ✅            | ❌           |
| **Essence**  | ✅             | ❌ **Never**    | ✅            | ✅           |

**MIND** and **ComiRec** both extract multi-interest representations and show strong recall on benchmark datasets — but they require training on the full population's interaction graph. Essence achieves multi-interest modelling within a single user's history, with no cross-user signal at any stage. This makes it trivially privacy-preserving: the system works entirely on-device with no data leaving the user's environment.

---

## Experimental Results

We tested Essence against two baselines on the **Last.fm-1K** dataset, evaluating across 99 users.

### Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Last.fm-1K (99 users, 50–500 interactions each) |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Split | Random 80/20 hold-out per user |
| Long-tail definition | Tracks heard by exactly 1 user (singletons = 89.8% of catalogue) |
| Evaluation | Recall@10 and Long-Tail Recall@10 |
| K (clusters) | 3 |
| M (recommendations) | 10 |

### The Three Systems

| ID | System | How it works |
|----|--------|--------------|
| **A** | CF Baseline | Recommends globally most-played tracks the user hasn't heard. Pure popularity. |
| **B** | Content Baseline | Embeds all the user's tracks and takes the average. Recommends items closest to this single "blurred" taste vector. |
| **C** | Essence | Clusters the user's embeddings into 3 profiles. Selects the profile matching recent listening. Recommends from that cluster. |

### Results

| System              | Recall@10 | LT-Recall@10 |
|:--------------------|:---------:|:------------:|
| CF (Popularity)     | 0.0081    | 0.0000       |
| Content (Avg Emb)   | 0.0414    | 0.0032       |
| **Essence (K=3)**   | **0.0616**| **0.0146**   |

### What the numbers mean

- **CF gets zero long-tail recall.** Popularity-ranked items are never singletons by definition, so CF provably cannot discover niche content. This confirms the core critique of CF.

- **Content is 5× better than CF on overall recall.** Semantic similarity to a mean embedding already beats pure popularity — evidence that embedding-based methods are worth using even in their simplest form.

- **Essence is 49% better than Content on overall recall** and **4.5× better on long-tail recall.** The active centroid — a focused, cluster-specific vector — surfaces niche tracks far more effectively than the blurry average embedding. This is the empirical proof of the paper's central claim: *depth over breadth requires separation of interests, not averaging them.*

---

## Repository Structure — Every File Explained

```
essence/
│
├── data/
│   ├── download_lastfm.py       ← Downloads Last.fm-1K from MTG mirror (or HuggingFace fallback).
│   │                               Extracts the raw .tsv file into data/raw/.
│   │
│   ├── preprocess.py            ← Cleans the raw data: drops nulls, deduplicates (user, track)
│   │                               pairs, filters users to 50–500 interactions, builds item_text
│   │                               (track + artist), defines long-tail items (play count = 1),
│   │                               and produces an 80/20 random hold-out split per user.
│   │                               Outputs: train_interactions.pkl, test_interactions.pkl,
│   │                                        long_tail_ids.pkl
│   │
│   ├── raw/                     ← Extracted Last.fm-1K dataset files (gitignored — ~1.5 GB).
│   │                               Contains userid-timestamp-artid-artname-traid-traname.tsv.
│   │
│   ├── train_interactions.pkl   ← 80% of each user's tracks (random hold-out). DataFrame with
│   │                               columns: user_id, timestamp, artist_id, artist_name,
│   │                               track_id, track_name, item_text. Generated at runtime.
│   │
│   ├── test_interactions.pkl    ← Held-out 20% of each user's tracks. Same schema as train.
│   │                               Generated at runtime.
│   │
│   └── long_tail_ids.pkl        ← Python set of track_ids with exactly 1 play in train.
│                                   These are the niche items used for LT-Recall@10.
│                                   Generated at runtime.
│
├── embeddings/
│   ├── generate_embeddings.py   ← Loads train + test interactions, encodes all unique tracks
│   │                               using all-MiniLM-L6-v2 (batch size 256), and saves a
│   │                               dict {track_id → np.ndarray(384,)} to item_embeddings.pkl.
│   │                               Embeds train ∪ test so every item is recoverable.
│   │
│   └── item_embeddings.pkl      ← Cached embedding map: 22,767 tracks × 384 dims.
│                                   ~35 MB. Generated at runtime (gitignored).
│
├── models/
│   └── recommenders.py          ← The three systems, importable as functions:
│                                   • cf_recommend()       — popularity baseline
│                                   • content_recommend()  — mean-embedding baseline
│                                   • essence_recommend()  — K-means active centroid
│                                   Also contains cosine_similarity() helper.
│
├── evaluation/
│   └── evaluate.py              ← Defines recall_at_k() and long_tail_recall_at_k() metrics.
│                                   Runs all three systems over N sampled users and saves
│                                   per-user results to a CSV. Supports --users, --K, --M,
│                                   --seed, --output flags.
│
├── results/
│   ├── evaluation_results.csv        ← Output from the first run (debugging reference).
│   └── evaluation_results_v2.csv     ← Output after bug fixes. This is the canonical result.
│
├── debug/
│   └── diagnose.py              ← Diagnostic script that prints dataset shapes, train/test/
│                                   embedding overlap statistics, and type checks. Useful for
│                                   verifying the data pipeline is working correctly.
│
├── notebooks/                   ← Scratch exploration space. Currently empty.
│
├── paper/                       ← Manuscript drafts. Currently empty.
│
├── run_experiment.py            ← End-to-end runner. Checks for cached .pkl files, auto-runs
│                                   preprocess and embedding steps if missing, then calls
│                                   evaluation and prints results. The single command to run
│                                   the full experiment. Supports: --users, --K, --M, --output.
│
├── requirements.txt             ← All Python dependencies. Install with: pip install -r requirements.txt
│
└── README.md                    ← This file.
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Mac or Linux
- ~4 GB free disk space (dataset ≈ 1.5 GB, embeddings ≈ 35 MB)
- No GPU required

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/biditdas18/essence
cd essence

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Experiment

```bash
# Step 1 — Download the Last.fm-1K dataset (~1.5 GB)
python data/download_lastfm.py

# Step 2 — Run the full experiment
# (preprocessing and embedding generation are auto-triggered if needed)
python run_experiment.py --users 200 --K 3 --M 10
```

Results print to terminal and save to `results/evaluation_results.csv`.

### Running Steps Individually

If you want to run each stage separately:

```bash
# Clean and split the raw data
python data/preprocess.py

# Generate embeddings (takes ~60 seconds on CPU)
python embeddings/generate_embeddings.py

# Run evaluation only
python evaluation/evaluate.py --users 200 --K 3 --M 10
```

### Running the Diagnostic

If something looks wrong, the diagnostic script will tell you:

```bash
python debug/diagnose.py
```

This prints train/test sizes, embedding map coverage, long-tail item counts, and type-checks all the data — everything you need to confirm the pipeline is healthy.

---

## Citation

If you use this code, please cite:

```
Das, B. (2025). Essence: A Personal Embedding Cluster Framework
for Depth-Optimized Recommendations. SSRN. [DOI TBD]
```

---

## References

- Cen, Y., et al. (2019). **MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall.** RecSys 2019.
- Cen, Y., et al. (2020). **ComiRec: Controllable Multi-Interest Framework for Recommendation.** KDD 2020.
- Sun, F., et al. (2019). **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.** CIKM 2019.
- Reimers, N. & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** EMNLP 2019.
