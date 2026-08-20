"""
experiments/movielens/fetch_tmdb_overviews.py
--------------------------------------------------
Step 11a.5: fetch plot overviews from TMDb's /movie/{id} endpoint for the
7,724-item candidate set (data/movielens_processed/item_candidates.csv).

Design constraints from the task:
  - Read TMDB_API_KEY from environment (.env, never hardcoded, never
    logged/printed).
  - Stay well under the 40 req/s TMDb limit -- paced at ~12 req/s.
  - Incremental checkpointing: append each result to a JSONL file
    immediately (flushed), not held in memory -- a crash partway
    through does not lose completed work.
  - Resumable: on start, skip any tmdbId already present in the
    checkpoint file.
  - Never let one bad response crash the run: log failures to a
    separate file and continue.

Outputs:
  data/movielens_processed/tmdb_overviews.jsonl   (one JSON object per
                                                    successfully fetched movie)
  data/movielens_processed/tmdb_failures.jsonl     (tmdbId, movieId, reason)

Run:
    python experiments/movielens/fetch_tmdb_overviews.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "movielens_processed"
ENV_PATH = BASE_DIR / ".env"

OVERVIEWS_PATH = PROC_DIR / "tmdb_overviews.jsonl"
FAILURES_PATH = PROC_DIR / "tmdb_failures.jsonl"

REQUESTS_PER_SECOND = 12
TIMEOUT_S = 10


def load_api_key():
    if not ENV_PATH.exists():
        raise RuntimeError(f".env not found at {ENV_PATH}")
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("TMDB_API_KEY not found in .env")


def load_already_fetched():
    fetched = set()
    if OVERVIEWS_PATH.exists():
        with open(OVERVIEWS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fetched.add(int(obj["tmdbId"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return fetched


def load_already_failed():
    failed = set()
    if FAILURES_PATH.exists():
        with open(FAILURES_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    failed.add(int(obj["tmdbId"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    return failed


def fetch_one(tmdb_id: int, api_key: str) -> dict:
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=en-US"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def main():
    api_key = load_api_key()
    print(f"[fetch_tmdb] API key loaded (length={len(api_key)}, not printed)")

    candidates = pd.read_csv(PROC_DIR / "item_candidates.csv")
    candidates = candidates.dropna(subset=["tmdbId"])
    candidates["tmdbId"] = candidates["tmdbId"].astype(int)
    total = len(candidates)
    print(f"[fetch_tmdb] {total:,} items with a tmdbId to fetch")

    already_fetched = load_already_fetched()
    already_failed = load_already_failed()
    print(f"[fetch_tmdb] Resuming: {len(already_fetched):,} already fetched, {len(already_failed):,} already failed")

    todo = candidates[~candidates["tmdbId"].isin(already_fetched | already_failed)]
    print(f"[fetch_tmdb] {len(todo):,} remaining to fetch")

    delay = 1.0 / REQUESTS_PER_SECOND
    n_success, n_fail = 0, 0
    t0 = time.time()

    with open(OVERVIEWS_PATH, "a") as f_ok, open(FAILURES_PATH, "a") as f_fail:
        for i, row in enumerate(todo.itertuples(), 1):
            t_req0 = time.time()
            try:
                data = fetch_one(row.tmdbId, api_key)
                record = {
                    "tmdbId": row.tmdbId,
                    "movieId": row.movieId,
                    "title": row.title,
                    "genres": row.genres,
                    "overview": data.get("overview", ""),
                    "tmdb_title": data.get("title", ""),
                }
                f_ok.write(json.dumps(record) + "\n")
                f_ok.flush()
                n_success += 1
            except urllib.error.HTTPError as e:
                f_fail.write(json.dumps({"tmdbId": row.tmdbId, "movieId": row.movieId, "reason": f"HTTPError {e.code}"}) + "\n")
                f_fail.flush()
                n_fail += 1
            except Exception as e:
                f_fail.write(json.dumps({"tmdbId": row.tmdbId, "movieId": row.movieId, "reason": f"{type(e).__name__}: {e}"}) + "\n")
                f_fail.flush()
                n_fail += 1

            if i % 200 == 0 or i == len(todo):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(todo)}] success={n_success} fail={n_fail} elapsed={elapsed:.0f}s")

            # rate limit pacing
            elapsed_req = time.time() - t_req0
            if elapsed_req < delay:
                time.sleep(delay - elapsed_req)

    print(f"\n[fetch_tmdb] Done. This run: {n_success} succeeded, {n_fail} failed.")
    print(f"[fetch_tmdb] Totals: {len(already_fetched) + n_success} fetched, {len(already_failed) + n_fail} failed, out of {total}")


if __name__ == "__main__":
    main()
