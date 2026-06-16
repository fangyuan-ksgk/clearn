"""clearn.fastweight — training-free distant memory as a linear-attention fast weight.

Founding intuition, made literal (Schlag et al. 2021): the distant KV cache is an
associative memory equivalent to a weight matrix of key->value outer products. So
instead of *learning* a LoRA to replace distant attention, we *compute* it in
closed form from the sequence's own KV:

  encode pass (full seq):  per layer/kv-head   S_h = Σ_{j distant} φ(k_j) ⊗ v_j   (φ=elu+1)
                                                z_h = Σ_{j distant} φ(k_j)
  score pass (window only): recent query q_i gets, on top of exact local softmax,
                            a linear-attention read of the distant memory:
                                N_dist_i = φ(q_i) · S_h ,  D_dist_i = φ(q_i) · z_h
                            o_i = (N_rec_i + λ N_dist_i) / (D_rec_i + λ N_dist_den_i)

Because S_h is a deterministic function of *this* sequence's distant KV, it is never
out-of-distribution: it generalizes to unseen sequences for free and degrades
gracefully. λ is a single global scalar (swept or lightly tuned) — no per-doc
training. This is the elegant complement to the learned hypernetwork.
"""
from __future__ import annotations
import torch
from transformers.models.qwen2 import modeling_qwen2 as _q

_orig_forward = _q.Qwen2Attention.forward


class FW:
    """Global controller toggled around forward passes."""
    mode = "off"          # "encode" | "score" | "off"
    K = 64
    lam = 1.0
    store: dict = {}      # layer_idx -> (S[n_kv,d,d], z[n_kv,d])

    @classmethod
    def reset(cls):
        cls.store = {}


def _phi(x):
    return torch.nn.functional.elu(x) + 1.0


def _patched_forward(self, hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, **kwargs):
    if FW.mode == "off":
        return _orig_forward(self, hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)

    B, T, _ = hidden_states.shape
    hs = (B, T, -1, self.head_dim)
    q = self.q_proj(hidden_states).view(hs).transpose(1, 2)        # [B,Hq,T,d]
    k = self.k_proj(hidden_states).view(hs).transpose(1, 2)        # [B,Hkv,T,d]
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _q.apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups

    if FW.mode == "encode":
        # capture distant memory from this layer, then run normal attention
        kd, vd = k[:, :, :T - FW.K, :].float(), v[:, :, :T - FW.K, :].float()  # [B,Hkv,Nd,d]
        pk = _phi(kd)
        S = torch.einsum("bhnd,bhne->bhde", pk, vd)[0]              # [Hkv,d,d]
        z = pk.sum(2)[0]                                            # [Hkv,d]
        FW.store[self.layer_idx] = (S, z)
        return _orig_forward(self, hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)

    # FW.mode == "score": convex blend of normalized recent-softmax and normalized
    # distant linear-attention reads -> o = (1-a) o_rec + a o_dist  (a = FW.lam in [0,1])
    S, z = FW.store[self.layer_idx]                                # [Hkv,d,d],[Hkv,d]
    qf = q.float()
    kf = _q.repeat_kv(k, g).float()
    vf = _q.repeat_kv(v, g).float()
    Lq = q.shape[2]
    scores = torch.matmul(qf, kf.transpose(2, 3)) * self.scaling   # [B,Hq,Lq,Lq]
    neg = torch.finfo(scores.dtype).min
    causal = torch.tril(torch.ones(Lq, Lq, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, neg)
    o_rec = torch.matmul(torch.softmax(scores, dim=-1), vf)        # [B,Hq,Lq,d] normalized
    # normalized distant linear read (per kv-head memory repeated to query heads)
    Sr = S.repeat_interleave(g, dim=0)                            # [Hq,d,d]
    zr = z.repeat_interleave(g, dim=0)                            # [Hq,d]
    pq = _phi(qf)                                                  # [B,Hq,Lq,d]
    N_dist = torch.einsum("bhld,hde->bhle", pq, Sr)              # [B,Hq,Lq,d]
    D_dist = torch.einsum("bhld,hd->bhl", pq, zr).unsqueeze(-1)   # [B,Hq,Lq,1]
    o_dist = N_dist / (D_dist + 1e-6)
    a = FW.lam
    if torch.is_tensor(a) and a.ndim > 0:          # per-layer gate vector
        a = a[self.layer_idx]
    o = (1.0 - a) * o_rec + a * o_dist
    o = o.to(q.dtype).transpose(1, 2).reshape(B, T, -1).contiguous()
    return self.o_proj(o), None


def enable():
    _q.Qwen2Attention.forward = _patched_forward


def disable():
    _q.Qwen2Attention.forward = _orig_forward
    FW.mode = "off"
