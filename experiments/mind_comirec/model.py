"""
experiments/mind_comirec/model.py
------------------------------------
Hand-implemented MIND (Li et al., CIKM 2019) and ComiRec-SA (Cen et al.,
RecSys 2020) in PyTorch. RecBole does not ship either model (verified
against PyPI 1.2.1 and the GitHub master branch), so these are built
directly from the papers' architecture descriptions.

Both models share:
  - an item embedding table (optionally initialized from this repo's
    existing pretrained sentence-transformer item embeddings, see
    "Deviations from the papers" below)
  - a multi-interest extraction module (dynamic routing for MIND,
    self-attention for ComiRec-SA) producing K interest vectors per user
  - label-aware attention at training time to pick a target-aware
    combination of interests for the sampled-softmax loss
  - full-catalog serving: max-similarity over the K interest vectors
    against all unseen candidate items (standard multi-interest retrieval)

Deviations from the papers (disclosed, not hidden)
----------------------------------------------------
1. Embedding dim = 384 and initialized from this repo's existing
   sentence-transformer embeddings (all-MiniLM-L6-v2), rather than a
   smaller dim learned from scratch. The papers train on populations of
   millions of interactions; Last.fm-1K here has only 99 users / ~21K
   train interactions, far too little to learn a useful embedding table
   from a random init. Content-informed initialization is used so the
   comparison isn't crippled by this repo's data scale — but it means
   these models get a head start MIND/ComiRec's original setup did not
   have. Flagged for the verification step.
2. MIND's adaptive interest-number heuristic (K' = max(1, min(K,
   log2(|history|)))) is NOT implemented — fixed K interests for every
   user regardless of history length.
   ponytail: ceiling = short-history users get the same K as long-history
   users, which the paper's adaptive-K avoids. Upgrade path: mask out
   the lowest-magnitude interest capsules per user at eval time using
   the paper's log2 heuristic if this turns out to matter.
3. Negative sampling is uniform random over the full item vocabulary
   (not popularity-weighted as commonly done in these papers).
   ponytail: ceiling = slightly easier negatives than popularity-based
   sampling would give. Upgrade path: multinomial sampling weighted by
   train-set item frequency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0


def squash(s: torch.Tensor, dim=-1, eps=1e-8) -> torch.Tensor:
    """Capsule squashing nonlinearity: ||s||^2/(1+||s||^2) * s/||s||"""
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * s / torch.sqrt(sq_norm + eps)


class MultiInterestBase(nn.Module):
    def __init__(self, n_items: int, embedding_dim: int, K: int,
                item_embedding_init: np.ndarray = None, max_seq_len: int = 50):
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=PAD_IDX)
        if item_embedding_init is not None:
            with torch.no_grad():
                self.item_embedding.weight[1:] = torch.from_numpy(item_embedding_init).float()

    def extract_interests(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item indices (0 = pad). Returns (B, K, D) interest vectors."""
        raise NotImplementedError

    def label_aware_attention(self, interests: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        interests: (B, K, D), target_emb: (B, D).
        Returns a target-aware user vector (B, D): weighted combination of
        interest capsules, weighted by similarity to the target item
        (Li et al. 2019, Eq. 12; power-law sharpening approximated with a
        temperature-scaled softmax here).
        """
        scores = torch.einsum("bkd,bd->bk", interests, target_emb)
        weights = F.softmax(scores * 10.0, dim=-1)  # temperature sharpens toward the best-matching capsule
        return torch.einsum("bk,bkd->bd", weights, interests)

    def forward(self, seq: torch.Tensor, target: torch.Tensor = None):
        interests = self.extract_interests(seq)
        if target is None:
            return interests
        target_emb = self.item_embedding(target)
        return self.label_aware_attention(interests, target_emb)


class MIND(MultiInterestBase):
    """Multi-Interest Network with Dynamic routing (Li et al., CIKM 2019)."""

    def __init__(self, n_items, embedding_dim, K, item_embedding_init=None,
                max_seq_len=50, routing_iters=3):
        super().__init__(n_items, embedding_dim, K, item_embedding_init, max_seq_len)
        self.routing_iters = routing_iters
        # Shared bilinear mapping matrix S (MIND simplifies standard CapsNet
        # by sharing one S across all input->output capsule pairs).
        self.S = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.05)

    def extract_interests(self, seq: torch.Tensor) -> torch.Tensor:
        mask = (seq != PAD_IDX).float()  # (B, L)
        item_emb = self.item_embedding(seq)  # (B, L, D)
        B, L, D = item_emb.shape
        K = self.K

        u_hat = torch.einsum("bld,de->ble", item_emb, self.S)  # (B, L, D)

        b = torch.zeros(B, L, K, device=seq.device)
        neg_inf_mask = (mask == 0).unsqueeze(-1)  # (B, L, 1)

        v = None
        for it in range(self.routing_iters):
            b_masked = b.masked_fill(neg_inf_mask, -1e9)
            c = F.softmax(b_masked, dim=2)  # (B, L, K), softmax over interest capsules
            c = c * mask.unsqueeze(-1)
            s = torch.einsum("blk,bld->bkd", c, u_hat)  # (B, K, D)
            v = squash(s, dim=-1)  # (B, K, D)
            if it < self.routing_iters - 1:
                agreement = torch.einsum("bld,bkd->blk", u_hat, v)
                b = b + agreement
        return v


class ComiRecSA(MultiInterestBase):
    """ComiRec, self-attentive variant (Cen et al., RecSys 2020)."""

    def __init__(self, n_items, embedding_dim, K, item_embedding_init=None,
                max_seq_len=50, d_hidden=None):
        super().__init__(n_items, embedding_dim, K, item_embedding_init, max_seq_len)
        d_hidden = d_hidden or embedding_dim
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.W1 = nn.Linear(embedding_dim, d_hidden, bias=False)
        self.W2 = nn.Linear(d_hidden, K, bias=False)

    def extract_interests(self, seq: torch.Tensor) -> torch.Tensor:
        mask = (seq != PAD_IDX)  # (B, L) bool
        item_emb = self.item_embedding(seq)  # (B, L, D)
        B, L, D = item_emb.shape

        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = item_emb + self.pos_embedding(positions)

        scores = self.W2(torch.tanh(self.W1(h)))  # (B, L, K)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        A = F.softmax(scores, dim=1)  # softmax over sequence positions, per interest column

        interests = torch.einsum("blk,bld->bkd", A, item_emb)  # (B, K, D)
        return interests


MODELS = {"mind": MIND, "comirec": ComiRecSA}
