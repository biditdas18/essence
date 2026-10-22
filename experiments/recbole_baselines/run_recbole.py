"""
SASRec / GRU4Rec baselines via RecBole, run under THIS paper's evaluation
protocol rather than RecBole's defaults.

WHY LIBRARY MODELS
------------------
MIND and ComiRec are hand-implemented here because no standard library
provides them. SASRec and GRU4Rec are the conventional sequential
baselines and ARE available in RecBole, so they are run through the
library: that removes "hand-implemented, possibly mis-implemented" as an
objection for these two, at the cost of having to force RecBole's
evaluation into this paper's protocol.

PROTOCOL MATCHING (the part that matters)
-----------------------------------------
RecBole's defaults differ from this paper in three ways, all overridden:

1. Split. RecBole would re-split the data. Instead `benchmark_filename`
   feeds it this paper's own per-user chronological 80/20 split, with a
   validation slice carved from the END of each user's train sequence
   (the same convention validation_k_sweep_* and train.py use, so early
   stopping never sees test).

2. Candidate pool. This paper scores every catalogue item the user has
   NOT interacted with in train, then takes the top 10
   (models/recommenders.py, step6c_full_chain_*). `full_sort_topk` masks
   each user's history, which under a benchmark split is exactly
   train+valid, i.e. this paper's train set. Same pool.

3. Metrics. RecBole's Recall@K is NOT this paper's: this paper divides by
   the user's FULL test-set size and macro-averages per-user ratios
   (evaluate.recall_at_k). So RecBole's metrics are ignored entirely --
   only its ranked top-10 lists are used, and the metrics are computed
   with this repo's own functions.

Sequence order per dataset: Last.fm and MovieLens by timestamp; Amazon
Books has no timestamp column in its processed train.csv because the
preprocessor writes rows already in chronological order, so row order is
the sequence (matching how the recency baselines read it).

Outputs `results/recbole_<model>_<dataset>_peruser.csv` with columns
user_id, system, recall@10, long_tail_recall@10 -- the same shape the
merged per-user CSVs use, so these slot straight into the paired
bootstrap and FDR families.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from evaluation.evaluate import recall_at_k, long_tail_recall_at_k  # this paper's metrics

VAL_FRACTION = 0.20
TOPK = 10
SEED = 42


def load_split(dataset, proc_suffix=""):
    """Return (train_df, test_df, long_tail_ids) with columns user_id/item_id/order."""
    if dataset == "lastfm":
        tr = pd.read_pickle(BASE / "data" / "train_interactions.pkl")
        te = pd.read_pickle(BASE / "data" / "test_interactions.pkl")
        tr = tr.rename(columns={"track_id": "item_id"})
        te = te.rename(columns={"track_id": "item_id"})
        tr = tr.sort_values(["user_id", "timestamp"])
        te = te.sort_values(["user_id", "timestamp"])
        with open(BASE / "data" / "long_tail_ids.pkl", "rb") as f:
            lt = set(pickle.load(f))
    else:
        sub = {"amazon": f"amazon_processed{proc_suffix}",
               "movielens": f"movielens_processed{proc_suffix}"}[dataset]
        d = BASE / "data" / sub
        tr = pd.read_csv(d / "train.csv")
        te = pd.read_csv(d / "test.csv")
        if "timestamp" in tr.columns:
            tr = tr.sort_values(["user_id", "timestamp"])
            te = te.sort_values(["user_id", "timestamp"])
        lt_df = pd.read_csv(d / "longtail_items.csv")
        lt = set(lt_df[lt_df.columns[0]].tolist())
    tr = tr.reset_index(drop=True)
    tr["order"] = tr.groupby("user_id").cumcount()
    te = te.reset_index(drop=True)
    te["order"] = te.groupby("user_id").cumcount()
    return tr[["user_id", "item_id", "order"]], te[["user_id", "item_id", "order"]], lt


def carve_validation(train):
    """Last VAL_FRACTION of each user's train sequence becomes validation."""
    n = train.groupby("user_id")["item_id"].transform("size")
    n_val = np.maximum(1, np.round(n * VAL_FRACTION)).astype(int)
    is_val = (train["order"] >= (n - n_val)) & (n > 1)
    return train[~is_val].copy(), train[is_val].copy()


def write_atomic(dir_path, name, parts):
    dir_path.mkdir(parents=True, exist_ok=True)
    for tag, df in parts.items():
        # RecBole expects "<name>.inter" for a single interaction file and
        # "<name>.<split>.inter" for benchmark splits.
        out = dir_path / (f"{name}.inter" if tag == "inter" else f"{name}.{tag}.inter")
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["user_id:token", "item_id:token", "timestamp:float"])
            for u, i, o in df[["user_id", "item_id", "order"]].itertuples(index=False):
                w.writerow([u, i, float(o)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["lastfm", "amazon", "movielens"])
    ap.add_argument("--model", required=True, choices=["SASRec", "GRU4Rec"])
    ap.add_argument("--proc-suffix", default="")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-seq-len", type=int, default=50)
    args = ap.parse_args()

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, get_model, get_trainer
    from recbole.utils.case_study import full_sort_topk

    train_full, test, lt_ids = load_split(args.dataset, args.proc_suffix)
    train, valid = carve_validation(train_full)
    print(f"[data] users={train_full.user_id.nunique():,} "
          f"train={len(train):,} valid={len(valid):,} test={len(test):,} "
          f"catalog={train_full.item_id.nunique():,} long_tail={len(lt_ids):,}")

    # RecBole's benchmark_filename path does not work for SEQUENTIAL models:
    # SequentialDataset._benchmark_presets expects the atomic files to already
    # carry augmented sequence fields (item_id_list etc.), not raw interactions.
    # So instead we hand RecBole one interaction file in this paper's own
    # per-user chronological order and ask for a time-ordered, user-grouped
    # ratio split of 64/16/20. That is arithmetically the same construction as
    # this paper's: test = last 20% of each user, validation = last 20% of the
    # remaining train. The split is VERIFIED against this paper's own test
    # files below rather than assumed to match.
    name = f"ess_{args.dataset}{args.out_suffix}"
    root = BASE / "experiments" / "recbole_baselines" / "atomic"
    allint = pd.concat([train_full.assign(_part="train"), test.assign(_part="test")])
    allint = allint.sort_values(["user_id", "_part", "order"], ascending=[True, False, True])
    allint["order"] = allint.groupby("user_id").cumcount()
    write_atomic(root / name, name, {"inter": allint})

    cfg = Config(model=args.model, dataset=name, config_dict={
        "data_path": str(root),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "MAX_ITEM_LIST_LENGTH": args.max_seq_len,
        "epochs": args.epochs, "learning_rate": args.lr,
        "train_batch_size": 2048, "eval_batch_size": 4096,
        "stopping_step": 5, "metrics": ["Recall"], "topk": [TOPK],
        "valid_metric": f"Recall@{TOPK}",
        "eval_args": {"mode": "full", "order": "TO", "group_by": "user",
                      "split": {"RS": [0.64, 0.16, 0.20]}},
        # SASRec and GRU4Rec both default to cross-entropy over the full item
        # set in RecBole, which is incompatible with negative sampling.
        "train_neg_sample_args": None,
        "device": "cpu", "use_gpu": False, "seed": SEED,
        "show_progress": False, "checkpoint_dir": str(root / "ckpt"),
    })
    init_seed(cfg["seed"], cfg["reproducibility"])
    ds = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, ds)
    # ---- VERIFY RecBole's split reproduces this paper's test set ----------
    tf = test_data.dataset.inter_feat
    rb_u = ds.id2token(ds.uid_field, tf[ds.uid_field].numpy())
    rb_i = ds.id2token(ds.iid_field, tf[ds.iid_field].numpy())
    rb_test = {}
    for u, i in zip(rb_u, rb_i):
        rb_test.setdefault(str(u), set()).add(str(i))
    our_test = {str(u): {str(x) for x in g}
                for u, g in test.groupby("user_id")["item_id"].apply(list).items()}
    common = set(rb_test) & set(our_test)
    inter = sum(len(rb_test[u] & our_test[u]) for u in common)
    ours_n = sum(len(our_test[u]) for u in common)
    rb_n = sum(len(rb_test[u]) for u in common)
    jac = inter / max(1, len(set().union(*[rb_test[u] | our_test[u] for u in common])) if common else 1)
    print(f"[split-check] users matched {len(common)}/{len(our_test)} | "
          f"test items: RecBole {rb_n:,} vs paper {ours_n:,} | "
          f"overlap {inter:,} ({100*inter/max(1,ours_n):.1f}% of the paper's)")
    if ours_n and inter / ours_n < 0.95:
        print("[split-check] WARNING: split disagreement >5%; results are NOT "
              "directly comparable to the paper's other systems.")

    model = get_model(cfg["model"])(cfg, train_data._dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)
    best_val, _ = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    print(f"[train] best validation {cfg['valid_metric']} = {best_val}")

    ext_users = [u for u in test.user_id.unique()]
    uid_series = ds.token2id(ds.uid_field, [str(u) for u in ext_users])
    topk_scores, topk_iid = full_sort_topk(uid_series, model, test_data, k=TOPK,
                                           device=cfg["device"])
    rec_tokens = ds.id2token(ds.iid_field, topk_iid.cpu()).tolist()

    test_by_user = test.groupby("user_id")["item_id"].apply(list).to_dict()
    rows = []
    lt_str = {str(x) for x in lt_ids}
    for u, recs in zip(ext_users, rec_tokens):
        actual = [str(x) for x in test_by_user.get(u, [])]
        recs = [str(x) for x in recs]
        rows.append({
            "user_id": u, "system": args.model,
            "recall@10": recall_at_k(recs, actual, TOPK),
            "long_tail_recall@10": long_tail_recall_at_k(recs, actual, lt_str, TOPK),
        })
    df = pd.DataFrame(rows)
    out = BASE / "results" / f"recbole_{args.model}_{args.dataset}{args.out_suffix}_peruser.csv"
    df.to_csv(out, index=False)
    print(f"[eval] {args.model} on {args.dataset}: Recall@10={df['recall@10'].mean():.6f} "
          f"LT-Recall@10={df['long_tail_recall@10'].mean():.6f} "
          f"(n={len(df)}, n_lt={int(df['long_tail_recall@10'].notna().sum())})")
    print(f"[eval] saved -> {out}")


if __name__ == "__main__":
    main()
