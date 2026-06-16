"""clearn.hypernet — the hypernetwork that turns distant context into a LoRA.

Pipeline (Doc-to-LoRA / Generative-Adapter style, specialized to KV eviction):

  distant prefix tokens
        │  (frozen base model, one forward pass — detached), subsampled to M_ctx
        ▼
  hidden states H  ∈ [B, M_ctx, d]
        │  Perceiver cross-attention: r learnable queries pull from H
        ▼
  r latent vectors per item  z ∈ [B, r, d_h]   (one latent per LoRA rank component)
        │  shared conditioned generator g(z, emb[layer,module])
        ▼
  for every wrapped Linear:  A ∈ [B, r, in],  B ∈ [B, out, r]   (per-sample LoRA)

Batch-native: one hypernetwork forward produces a *different* LoRA for every item,
so a whole minibatch of documents can be scored in a single batched model forward
(the per-sample factors flow through LoRALinear's broadcast matmul). The generator
MLPs are shared across all layers/modules and conditioned on a learned
per-(layer,module) embedding, so adding layers costs only an embedding row.
"""
from __future__ import annotations
import math, torch, torch.nn as nn


class Perceiver(nn.Module):
    """r learnable queries cross-attend over the distant hidden states. Batched."""
    def __init__(self, d_model, d_h, r, n_heads=4):
        super().__init__()
        self.r, self.d_h, self.n_heads = r, d_h, n_heads
        self.in_ln = nn.LayerNorm(d_model)
        self.q = nn.Parameter(torch.randn(r, d_h) * (1 / math.sqrt(d_h)))
        self.k_proj = nn.Linear(d_model, d_h)
        self.v_proj = nn.Linear(d_model, d_h)
        self.ln = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(nn.Linear(d_h, 2 * d_h), nn.GELU(), nn.Linear(2 * d_h, d_h))

    def forward(self, H):  # H: [B, N, d_model]
        H = self.in_ln(H)
        B, N, _ = H.shape
        h, r, d_h = self.n_heads, self.r, self.d_h
        dk = d_h // h
        K = self.k_proj(H).view(B, N, h, dk).transpose(1, 2)   # [B,h,N,dk]
        V = self.v_proj(H).view(B, N, h, dk).transpose(1, 2)   # [B,h,N,dk]
        q = self.q.view(1, r, h, dk).transpose(1, 2).expand(B, h, r, dk)
        att = torch.softmax(q @ K.transpose(-1, -2) / math.sqrt(dk), dim=-1)  # [B,h,r,N]
        z = (att @ V).transpose(1, 2).reshape(B, r, d_h)        # [B,r,d_h]
        z = self.ln(z)
        return z + self.ff(z)


class HyperLoRA(nn.Module):
    def __init__(self, wrapped, d_model, r, d_h=256, emb_dim=64, init_scale=1e-3):
        super().__init__()
        self.r = r
        self.shapes = [(w.in_f, w.out_f) for w in wrapped]
        self.max_out = max(o for _, o in self.shapes)
        self.max_in = max(i for i, _ in self.shapes)
        self.perceiver = Perceiver(d_model, d_h, r)
        self.emb = nn.Embedding(len(wrapped), emb_dim)
        self.gen_A = nn.Sequential(nn.Linear(d_h + emb_dim, d_h), nn.GELU(), nn.Linear(d_h, self.max_in))
        self.gen_B = nn.Sequential(nn.Linear(d_h + emb_dim, d_h), nn.GELU(), nn.Linear(d_h, self.max_out))
        for g in (self.gen_A, self.gen_B):           # small-init head -> initial delta ~ 0
            nn.init.normal_(g[-1].weight, std=init_scale)
            nn.init.zeros_(g[-1].bias)

    def forward(self, H):
        """H: [B, N, d] (or [N, d] -> treated as B=1). Returns list of (A[B,r,in], B[B,out,r]).

        Vectorized: all M=(layers x targets) modules are generated in two batched
        MLP calls (over a [B, M, r, *] tensor) rather than a Python loop, so cost is
        independent of depth. The remaining loop only slices (no compute).
        """
        if H.dim() == 2:
            H = H.unsqueeze(0)
        B, M, r = H.shape[0], len(self.shapes), self.r
        z = self.perceiver(H)                                       # [B,r,d_h]
        zc = z.unsqueeze(1).expand(B, M, r, z.shape[-1])            # [B,M,r,d_h]
        Ec = self.emb.weight.view(1, M, 1, -1).expand(B, M, r, -1)  # [B,M,r,emb]
        cond = torch.cat([zc, Ec], dim=-1)
        A_all = self.gen_A(cond)                                    # [B,M,r,max_in]
        B_all = self.gen_B(cond)                                    # [B,M,r,max_out]
        return [(A_all[:, m, :, :i], B_all[:, m, :, :o].transpose(1, 2))
                for m, (i, o) in enumerate(self.shapes)]


def _subsample_distant(H_full, K, m_ctx):
    """H_full:[...,T,d] -> distant prefix subsampled to m_ctx tokens along time."""
    T = H_full.shape[-2]
    H = H_full[..., : T - K, :]
    Nd = H.shape[-2]
    if Nd > m_ctx:
        idx = torch.linspace(0, Nd - 1, m_ctx, device=H.device).long()
        H = H.index_select(-2, idx)
    return H.float().detach()


@torch.no_grad()
def encode_distant(model, ids, K, layer=-1, m_ctx=256):
    """Frozen forward; return subsampled distant hidden states (last layer).

    Uses the base transformer's last_hidden_state (no LM head, no all-layer
    stack) when layer==-1 — much cheaper for bulk precompute.
    """
    if layer == -1:
        H_full = model.model(input_ids=ids).last_hidden_state    # [1,T,d]
    else:
        H_full = model(input_ids=ids, output_hidden_states=True).hidden_states[layer]
    return _subsample_distant(H_full, K, m_ctx)[0]               # [m_ctx, d]


@torch.no_grad()
def prep_batch(model, ids_batch, K, m_ctx=256):
    """One base forward per item -> (teacher_lp[B,K,V], teacher_ce[B], H[B,m_ctx,d]).

    Skips the full [B,T,V] logits tensor: lm_head is applied only to the K scoring
    positions, and H reuses the same last_hidden_state. ~8x cheaper than separate
    full forwards for logits / loss / hidden states.
    """
    T = ids_batch.shape[1]
    base = model.model(input_ids=ids_batch).last_hidden_state         # [B,T,d]
    logitsK = model.lm_head(base[:, T - K - 1: T - 1, :]).float()     # [B,K,V]
    lp = logitsK.log_softmax(-1)
    tgt = ids_batch[:, T - K: T]                                      # [B,K]
    ce = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(1)        # [B]
    H = _subsample_distant(base, K, m_ctx)                           # [B,m_ctx,d]
    return lp, ce, H


def apply_lora(wrapped, factors):
    for w, (A, B) in zip(wrapped, factors):
        w.set_delta(A, B)


def clear_lora(wrapped):
    for w in wrapped:
        w.clear()
