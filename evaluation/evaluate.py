"""
evaluation/evaluate.py
-----------------------
Runs the three recommendation systems over a sample of users and
computes Recall@10 and Long-Tail Recall@10.

Outputs
-------
  results/evaluation_results.csv
  Aggregated results table printed to stdout

Run:
    python evaluation/evaluate.py              # uses defaults
    python evaluation/evaluate.py --users 200 --K 3 --M 10
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.recommenders import cf_recommend, content_recommend, essence_recommend

DATA_DIR       = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_DIR    = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_CSV_DEFAULT = RESULTS_DIR / "evaluation_results.csv"


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(recommended: list, actual: list, k: int = 10) -> float:
    """
    Of items user actually interacted with in the test set,
    what fraction appear in the top-K recommendations?
    """
    if not actual:
        return 0.0
    hits = len(set(recommended[:k]) & set(actual))
    return hits / min(len(actual), k)


def long_tail_recall_at_k(recommended: list, actual: list,
                           long_tail_ids: set, k: int = 10):
    """
    Same as recall@k but restricted to long-tail items only.
    Returns None if the user has no long-tail items in their test set
    (those users are excluded from the LT-Recall average).
    """
    actual_lt = [i for i in actual if i in long_tail_ids]
    if not actual_lt:
        return None  # skip users with no long-tail test items
    hits = len(set(recommended[:k]) & set(actual_lt))
    return hits / min(len(actual_lt), k)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(n_users: int = 200, K: int = 3, M: int = 10,
                   seed: int = 42,
                   output_csv: Path = None) -> pd.DataFrame:
    # Load data
    print("[eval] Loading train/test interactions …")
    train_df = pd.read_pickle(DATA_DIR / "train_interactions.pkl")
    test_df  = pd.read_pickle(DATA_DIR / "test_interactions.pkl")

    print("[eval] Loading item embeddings …")
    with open(EMBEDDINGS_DIR / "item_embeddings.pkl", "rb") as fh:
        item_embedding_map = pickle.load(fh)

    print("[eval] Loading long-tail item IDs …")
    with open(DATA_DIR / "long_tail_ids.pkl", "rb") as fh:
        long_tail_ids = pickle.load(fh)

    # Sample users present in both train and test
    all_users = list(
        set(train_df["user_id"].unique()) & set(test_df["user_id"].unique())
    )
    rng = np.random.default_rng(seed)
    sampled_users = rng.choice(
        all_users,
        size=min(n_users, len(all_users)),
        replace=False,
    ).tolist()

    print(f"[eval] Evaluating {len(sampled_users)} users "
          f"(K={K}, M={M}, seed={seed}) …\n")

    rows = []

    for user_id in tqdm(sampled_users, desc="Users"):
        actual = test_df[test_df["user_id"] == user_id]["track_id"].tolist()

        systems = {
            "CF (Popularity)":   cf_recommend(user_id, train_df, M),
            "Content (Avg Emb)": content_recommend(user_id, train_df,
                                                    item_embedding_map, M),
            f"Essence (K={K})":  essence_recommend(user_id, train_df,
                                                    item_embedding_map, K, M),
        }

        for system_name, recs in systems.items():
            r10   = recall_at_k(recs, actual, k=M)
            lt_r10 = long_tail_recall_at_k(recs, actual, long_tail_ids, k=M)

            rows.append({
                "user_id":            user_id,
                "system":             system_name,
                "recall@10":          r10,
                "long_tail_recall@10": lt_r10,   # may be None
            })

    results_df = pd.DataFrame(rows)
    out_path = output_csv if output_csv is not None else RESULTS_CSV_DEFAULT
    results_df.to_csv(out_path, index=False)
    print(f"\n[eval] Results saved to {out_path}")
    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print aggregated results table."""
    systems = results_df["system"].unique()

    header  = f"{'System':<22} | {'Recall@10':>10} | {'LT-Recall@10':>13}"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for system in systems:
        sub = results_df[results_df["system"] == system]
        r10    = sub["recall@10"].mean()
        lt_sub = sub["long_tail_recall@10"].dropna()
        lt_r10 = lt_sub.mean() if len(lt_sub) > 0 else float("nan")
        print(f"{system:<22} | {r10:>10.4f} | {lt_r10:>13.4f}")

    print(divider)
    lt_coverage = results_df[results_df["system"] == systems[0]]["long_tail_recall@10"].notna().sum()
    print(f"\nNote: LT-Recall averaged over {lt_coverage} users "
          f"with ≥1 long-tail test item.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Essence experiment")
    parser.add_argument("--users", type=int, default=200,
                        help="Number of users to sample (default: 200)")
    parser.add_argument("--K", type=int, default=3,
                        help="K-means clusters for Essence (default: 3)")
    parser.add_argument("--M", type=int, default=10,
                        help="Recommendation list length (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename (default: evaluation_results.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = RESULTS_DIR / args.output if args.output else None
    results_df = run_evaluation(
        n_users=args.users,
        K=args.K,
        M=args.M,
        seed=args.seed,
        output_csv=out,
    )
    print_summary(results_df)
