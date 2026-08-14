"""
experiments/mind_comirec/train.py
------------------------------------
Trains MIND or ComiRec-SA on Last.fm-1K or Amazon Books, using the exact
existing temporal train/test split (via dataset.py) and the exact
existing full-catalog ranking protocol (same candidate pool, same
recall_at_k / long_tail_recall_at_k functions imported from
evaluation/evaluate.py — not RecBole's or any other metric implementation).

Fixed, undtuned hyperparameters (picked once from the papers' recommended
ranges, not grid-searched — see model.py's docstring for disclosed
deviations from the original papers):
  K = 4 interests, routing_iters = 3 (MIND), embedding_dim = 384
  (matches this repo's existing sentence-transformer cache),
  max_seq_len = 50, batch_size = 64, lr = 1e-3 (Adam), up to 50 epochs
  with early stopping (patience = 5) on last-train-item validation loss,
  100 uniform-random negatives per positive.

Run:
    python experiments/mind_comirec/train.py --dataset lastfm --model mind
    python experiments/mind_comirec/train.py --dataset amazon --model comirec
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_dataset
from model import MODELS, PAD_IDX
from evaluation.evaluate import recall_at_k, long_tail_recall_at_k

RESULTS_DIR = BASE_DIR / "results"

K = 4
EMBEDDING_DIM = 384
MAX_SEQ_LEN = 50
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5
N_NEGATIVES = 100
ROUTING_ITERS = 3
SEED = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vocab(item_embedding_map: dict):
    item_ids = sorted(item_embedding_map.keys())
    item2idx = {iid: i + 1 for i, iid in enumerate(item_ids)}  # 0 reserved for PAD
    idx2item = {i + 1: iid for i, iid in enumerate(item_ids)}
    emb_matrix = np.zeros((len(item_ids), EMBEDDING_DIM), dtype=np.float32)
    for i, iid in enumerate(item_ids):
        emb_matrix[i] = item_embedding_map[iid]
    return item_ids, item2idx, idx2item, emb_matrix


def encode_seq(seq_item_ids, item2idx, max_len=MAX_SEQ_LEN):
    idxs = [item2idx[i] for i in seq_item_ids if i in item2idx]
    idxs = idxs[-max_len:]  # keep most recent
    padded = idxs + [PAD_IDX] * (max_len - len(idxs))
    return padded, len(idxs)


class NextItemDataset(Dataset):
    """(prefix, target) pairs via sliding window, excluding the final train item (reserved for val)."""

    def __init__(self, train_sequences, item2idx, max_len=MAX_SEQ_LEN):
        self.examples = []
        for uid, seq in train_sequences.items():
            idxs = [item2idx[i] for i in seq if i in item2idx]
            if len(idxs) < 2:
                continue
            # exclude the last item as a prediction target during training (reserved for val)
            for t in range(1, len(idxs) - 1):
                prefix = idxs[max(0, t - max_len):t]
                target = idxs[t]
                self.examples.append((prefix, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, max_len=MAX_SEQ_LEN):
    seqs = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    for i, (prefix, target) in enumerate(batch):
        L = len(prefix)
        seqs[i, :L] = torch.tensor(prefix, dtype=torch.long)
        targets[i] = target
    return seqs, targets


def build_val_examples(train_sequences, item2idx, max_len=MAX_SEQ_LEN):
    examples = []
    for uid, seq in train_sequences.items():
        idxs = [item2idx[i] for i in seq if i in item2idx]
        if len(idxs) < 2:
            continue
        prefix = idxs[max(0, len(idxs) - 1 - max_len):len(idxs) - 1]
        target = idxs[-1]
        examples.append((prefix, target))
    return examples


def sampled_softmax_loss(model, seq, target, n_items, n_neg, device):
    B = seq.shape[0]
    interests = model.extract_interests(seq)  # (B, K, D)
    target_emb = model.item_embedding(target)  # (B, D)
    user_vec = model.label_aware_attention(interests, target_emb)  # (B, D)

    pos_score = (user_vec * target_emb).sum(-1, keepdim=True)  # (B, 1)

    neg_idx = torch.randint(1, n_items + 1, (B, n_neg), device=device)  # avoid PAD_IDX=0
    neg_emb = model.item_embedding(neg_idx)  # (B, N, D)
    neg_score = torch.einsum("bd,bnd->bn", user_vec, neg_emb)  # (B, N)

    logits = torch.cat([pos_score, neg_score], dim=1)  # (B, 1+N)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


def evaluate_loss(model, examples, n_items, device, batch_size=256):
    if not examples:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            seq, target = collate(batch)
            seq, target = seq.to(device), target.to(device)
            loss = sampled_softmax_loss(model, seq, target, n_items, N_NEGATIVES, device)
            total += loss.item() * len(batch)
            count += len(batch)
    model.train()
    return total / count


def normalized(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8
    return m / norms


def full_catalog_eval(model, dataset_name, train_sequences, test_items, long_tail_ids,
                      item_ids, item2idx, device, M=10):
    n_items = len(item_ids)
    with torch.no_grad():
        C = model.item_embedding.weight[1:].detach().cpu().numpy()  # (n_items, D), idx order matches item_ids
    C = normalized(C)
    C_t = torch.from_numpy(C).float().to(device)

    rows = []
    model.eval()
    eval_users = sorted(set(train_sequences.keys()) & set(test_items.keys()))
    with torch.no_grad():
        for uid in eval_users:
            train_seq = train_sequences[uid]
            padded, L = encode_seq(train_seq, item2idx)
            if L == 0:
                continue
            seq_t = torch.tensor([padded], dtype=torch.long, device=device)
            interests = model.extract_interests(seq_t)[0]  # (K, D)
            interests_np = normalized(interests.detach().cpu().numpy())
            interests_t = torch.from_numpy(interests_np).float().to(device)

            scores = (C_t @ interests_t.T).max(dim=1).values.cpu().numpy()  # (n_items,)
            scores = scores.copy()

            seen_idxs = [item2idx[i] - 1 for i in train_seq if i in item2idx]  # -1: item2idx is 1-based
            scores[seen_idxs] = -2.0

            top_idx = np.lexsort((np.arange(n_items), -scores))[:M]
            recs = [item_ids[idx] for idx in top_idx]

            actual = test_items[uid]
            r10 = recall_at_k(recs, actual, k=M)
            lt = long_tail_recall_at_k(recs, actual, long_tail_ids, k=M)
            rows.append({"user_id": uid, "recall@10": r10,
                        "long_tail_recall@10": "" if lt is None else lt})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lastfm", "amazon"], required=True)
    parser.add_argument("--model", choices=["mind", "comirec"], required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--embed-init", choices=["pretrained", "random"], default="pretrained",
                        help="pretrained: init from this repo's sentence-transformer cache (default). "
                             "random: N(0, 0.01) init, matching the papers' original from-scratch setup.")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--tag", type=str, default="",
                        help="Appended to system name/output filename for ablation runs "
                             "(e.g. 'randominit', 'seed1'). Empty = canonical run (default behavior).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"[train] dataset={args.dataset} model={args.model} device={device} "
          f"embed_init={args.embed_init} lr={args.lr} seed={args.seed} K={args.K} tag={args.tag!r}")

    print("[train] Loading data ...")
    item_embedding_map, train_sequences, test_items, long_tail_ids = load_dataset(args.dataset)
    item_ids, item2idx, idx2item, emb_matrix = build_vocab(item_embedding_map)
    n_items = len(item_ids)
    print(f"[train] n_items={n_items} n_users(train)={len(train_sequences)}")

    train_ds = NextItemDataset(train_sequences, item2idx)
    val_examples = build_val_examples(train_sequences, item2idx)
    print(f"[train] train examples={len(train_ds)} val examples={len(val_examples)}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    ModelCls = MODELS[args.model]
    init_matrix = None if args.embed_init == "random" else emb_matrix
    model_kwargs = dict(n_items=n_items, embedding_dim=EMBEDDING_DIM, K=args.K,
                        item_embedding_init=init_matrix, max_seq_len=MAX_SEQ_LEN)
    if args.model == "mind":
        model_kwargs["routing_iters"] = ROUTING_ITERS
    model = ModelCls(**model_kwargs).to(device)
    if args.embed_init == "random":
        with torch.no_grad():
            model.item_embedding.weight[1:].normal_(0.0, 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    t0 = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            loss = sampled_softmax_loss(model, seq, target, n_items, N_NEGATIVES, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        val_loss = evaluate_loss(model, val_examples, n_items, device)
        history.append((epoch, train_loss, val_loss))
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train] Done in {elapsed:.1f}s. Best val_loss={best_val:.4f} "
          f"at epoch {history[[h[2] for h in history].index(min(h[2] for h in history))][0]}")

    print("[train] Running full-catalog eval ...")
    rows = full_catalog_eval(model, args.dataset, train_sequences, test_items, long_tail_ids,
                             item_ids, item2idx, device)

    base_name = "MIND" if args.model == "mind" else "ComiRec (SA)"
    system_name = f"{base_name} ({args.tag})" if args.tag else base_name
    for r in rows:
        r["system"] = system_name

    r10_mean = np.mean([r["recall@10"] for r in rows])
    lt_vals = [r["long_tail_recall@10"] for r in rows if r["long_tail_recall@10"] != ""]
    lt_mean = np.mean(lt_vals) if lt_vals else float("nan")
    print(f"[train] {system_name} on {args.dataset}: Recall@10={r10_mean:.4f} "
          f"LT-Recall@10={lt_mean:.4f} (n={len(rows)}, n_lt={len(lt_vals)})")

    file_stem = f"mind_comirec_{args.tag}_{args.dataset}" if args.tag else f"mind_comirec_results_{args.dataset}"
    out_path = RESULTS_DIR / f"{file_stem}.csv"
    existing_rows = []
    if out_path.exists():
        import pandas as pd
        existing = pd.read_csv(out_path)
        existing = existing[existing["system"] != system_name]
        existing_rows = existing.to_dict("records")

    all_rows = existing_rows + [
        {"user_id": r["user_id"], "system": r["system"],
         "recall@10": r["recall@10"], "long_tail_recall@10": r["long_tail_recall@10"]}
        for r in rows
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "system", "recall@10", "long_tail_recall@10"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[train] Saved to {out_path}")

    summary_stem = f"mind_comirec_summary_{args.tag}_{args.dataset}" if args.tag else f"mind_comirec_summary_{args.dataset}"
    summary_path = RESULTS_DIR / f"{summary_stem}.csv"
    summary_rows = []
    if summary_path.exists():
        import pandas as pd
        existing_summary = pd.read_csv(summary_path)
        existing_summary = existing_summary[existing_summary["system"] != system_name]
        summary_rows = existing_summary.to_dict("records")
    summary_rows.append({
        "system": system_name, "dataset": args.dataset, "recall@10": r10_mean,
        "lt_recall@10": lt_mean, "n_users": len(rows), "n_lt_users": len(lt_vals),
        "train_seconds": elapsed, "epochs_run": history[-1][0], "best_val_loss": best_val,
        "lr": args.lr, "seed": args.seed, "K": args.K, "embed_init": args.embed_init,
    })
    # Union of fieldnames across all rows (old-schema rows saved before lr/seed/K/
    # embed_init were added won't have those keys; DictWriter needs the full set).
    all_fieldnames = list(dict.fromkeys(k for row in summary_rows for k in row.keys()))
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[train] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
