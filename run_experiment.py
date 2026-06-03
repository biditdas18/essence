"""
run_experiment.py
-----------------
End-to-end runner for the Essence recommendation experiment.

Usage:
    python run_experiment.py                      # defaults
    python run_experiment.py --users 200 --K 3 --M 10

Pipeline:
    1. Preprocess raw Last.fm-1K data   (if not already done)
    2. Generate item embeddings          (if not already done)
    3. Evaluate all three systems on N sampled users
    4. Print results table
    5. Save results/evaluation_results.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_DIR       = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_script(script_path: Path):
    """Run a Python script as a subprocess, inheriting stdout/stderr."""
    print(f"\n{'='*60}")
    print(f"[runner] Running: {script_path.relative_to(BASE_DIR)}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print(f"\n[runner] ERROR: {script_path.name} exited with "
              f"code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full Essence recommendation experiment."
    )
    parser.add_argument("--users", type=int, default=200,
                        help="Number of users to evaluate (default: 200)")
    parser.add_argument("--K", type=int, default=3,
                        help="K-means clusters for Essence (default: 3)")
    parser.add_argument("--M", type=int, default=10,
                        help="Recommendation list length M (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for user sampling (default: 42)")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Output CSV filename in results/ (default: evaluation_results.csv)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  ESSENCE — Recommendation Experiment Runner")
    print("=" * 60)
    print(f"  Users   : {args.users}")
    print(f"  K       : {args.K}")
    print(f"  M       : {args.M}")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : results/{args.output}")

    # -----------------------------------------------------------------------
    # Step 1 — Preprocessing
    # -----------------------------------------------------------------------
    train_pkl = DATA_DIR / "train_interactions.pkl"
    if train_pkl.exists():
        print(f"\n[runner] train_interactions.pkl found — skipping preprocessing.")
    else:
        print("\n[runner] train_interactions.pkl not found — running preprocess.py …")
        run_script(DATA_DIR / "preprocess.py")

    # -----------------------------------------------------------------------
    # Step 2 — Embeddings
    # -----------------------------------------------------------------------
    embeddings_pkl = EMBEDDINGS_DIR / "item_embeddings.pkl"
    if embeddings_pkl.exists():
        print(f"\n[runner] item_embeddings.pkl found — skipping embedding generation.")
    else:
        print("\n[runner] item_embeddings.pkl not found — running generate_embeddings.py …")
        run_script(EMBEDDINGS_DIR / "generate_embeddings.py")

    # -----------------------------------------------------------------------
    # Step 3–5 — Evaluation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[runner] Running evaluation …")
    print(f"{'='*60}")

    # Import here so path setup in evaluate.py runs after subprocesses finish
    sys.path.insert(0, str(BASE_DIR))
    from evaluation.evaluate import run_evaluation, print_summary

    out_csv = BASE_DIR / "results" / args.output
    results_df = run_evaluation(
        n_users=args.users,
        K=args.K,
        M=args.M,
        seed=args.seed,
        output_csv=out_csv,
    )
    print_summary(results_df)

    print("\n[runner] Experiment complete.")
    print(f"[runner] Results saved to: results/{args.output}\n")


if __name__ == "__main__":
    main()
