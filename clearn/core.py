"""clearn.core — shared primitives for the online-KV-compression experiments.

The central idea: a long sequence is split into a *distant prefix* (the first
T-K tokens, whose KV cache we want to evict) and a *recent window* (the last K
tokens).  We measure the model's next-token loss on the recent window under
three regimes:

  * teacher    — full causal attention over all T tokens (KV fully kept)
  * evicted    — recent-window queries are blocked from attending to distant
                 keys (a pure local window of size K, optionally + attention sink)
  * adapted    — evicted attention, but with a (hypernetwork-generated) LoRA
                 active so the weights absorb the dropped context

Everything here is deliberately small and dependency-light so the demo notebook
can import it directly.
"""
from __future__ import annotations
import json, os, glob
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "data")


def load_model(name: str = MODEL_NAME, dtype=torch.bfloat16, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    return tok, model


def load_docs(config: str, n: int, min_chars: int = 6000):
    """Return up to `n` long `context` strings from a LongBench jsonl config."""
    path = os.path.join(DATA_DIR, f"{config}.jsonl")
    docs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line).get("context", "")
            if len(c) >= min_chars:
                docs.append(c)
            if len(docs) >= n:
                break
    return docs


def tokenize_fixed(tok, text: str, T: int, device="cuda"):
    ids = tok(text, return_tensors="pt").input_ids.to(device)[:, :T]
    return ids if ids.shape[1] == T else None


def many_sequences(tok, configs, T, max_total, per_doc=3, device="cuda"):
    """Build a pool of [1,T] sequences by slicing non-overlapping T-windows out of
    LongBench contexts (up to `per_doc` windows per context, for diversity)."""
    seqs = []
    for cfg, _ in configs:
        path = os.path.join(DATA_DIR, f"{cfg}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                c = json.loads(line).get("context", "")
                ids = tok(c, return_tensors="pt").input_ids.to(device)
                n = min(per_doc, ids.shape[1] // T)
                for w in range(n):
                    seqs.append(ids[:, w * T:(w + 1) * T])
                if len(seqs) >= max_total:
                    return seqs
    return seqs


def window_mask(T: int, K: int, evict: bool, sink: int = 0, device="cuda", dtype=torch.bfloat16):
    """Additive 4D attention mask [1,1,T,T].

    Lower-triangular causal mask; if `evict`, recent-window queries (index >= T-K)
    cannot attend to distant keys (index < T-K), except the first `sink` tokens
    (StreamingLLM-style attention sink) which are always visible.
    """
    minv = torch.finfo(dtype).min
    causal = torch.full((T, T), minv, device=device, dtype=dtype).triu(1)
    if evict:
        q = torch.arange(T, device=device)[:, None]
        k = torch.arange(T, device=device)[None, :]
        block = (q >= T - K) & (k < T - K)
        if sink > 0:
            block = block & (k >= sink)
        causal = causal.masked_fill(block, minv)
    return causal[None, None]


def window_loss(model, ids, K: int, mask=None, evict=False, sink=0, reduction="mean"):
    """Cross-entropy on the predictions of the last K tokens.

    Position T-K-1 predicts the first token of the recent window, etc.
    """
    T = ids.shape[1]
    if mask is None:
        mask = window_mask(T, K, evict=evict, sink=sink, device=ids.device, dtype=model.dtype)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
    )
    win = lp[T - K - 1:]
    return win.mean().item() if reduction == "mean" else win


def recent_window_loss(model, ids, K, reduction="mean"):
    """Evicted-regime loss via a *window-only* forward (no distant tokens processed).

    Under eviction, recent tokens attend only to recent keys, so their
    representations depend only on the recent window. We therefore forward just
    the last K+1 tokens with their true RoPE positions and score the predictions
    of the last K tokens. This is exactly equivalent to a full forward with the
    distant keys masked out (verified), but ~T/K times cheaper and is the
    faithful "distant KV is gone at inference" computation. LoRA, if injected,
    is active and absorbs the distant context.
    """
    T = ids.shape[1]
    win = ids[:, T - K - 1: T]                       # K+1 tokens
    pos = torch.arange(T - K - 1, T, device=ids.device)[None]
    out = model(input_ids=win, position_ids=pos)
    logits = out.logits[:, :-1, :].float()           # predict win[1:] -> last K tokens
    tgt = win[:, 1:]
    lp = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
    )
    return lp.mean() if reduction == "mean" else lp


def recent_window_logits(model, ids, K):
    """Log-probs on the last-K-token predictions via the window-only forward.

    Mirrors recent_window_loss but returns the [K, vocab] log-softmax (LoRA, if
    injected, is active). Used as the student in context distillation.
    """
    T = ids.shape[1]
    win = ids[:, T - K - 1: T]
    pos = torch.arange(T - K - 1, T, device=ids.device)[None]
    out = model(input_ids=win, position_ids=pos)
    return out.logits[0, :-1, :].float().log_softmax(-1)


def recent_window_logits_batch(model, ids_batch, K):
    """Batched window-only forward. ids_batch:[B,T] -> log-probs [B,K,vocab].

    A per-sample LoRA (different for each batch row) flows through the wrapped
    LoRALinear's broadcast matmul, so all docs are scored in one model call.
    """
    T = ids_batch.shape[1]
    win = ids_batch[:, T - K - 1: T]
    pos = torch.arange(T - K - 1, T, device=ids_batch.device)[None].expand(win.shape[0], -1)
    out = model(input_ids=win, position_ids=pos)
    return out.logits[:, :-1, :].float().log_softmax(-1)       # [B,K,vocab]


def recent_window_loss_batch(model, ids_batch, K):
    """Per-item evicted window CE. ids_batch:[B,T] -> [B] mean-CE per item."""
    T = ids_batch.shape[1]
    win = ids_batch[:, T - K - 1: T]
    pos = torch.arange(T - K - 1, T, device=ids_batch.device)[None].expand(win.shape[0], -1)
    logits = model(input_ids=win, position_ids=pos).logits[:, :-1, :].float()
    tgt = win[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
    ).view(win.shape[0], -1)
    return ce.mean(dim=1)


def teacher_window_loss(model, ids, K, reduction="mean"):
    """Full-KV loss on the same last-K-token predictions (the lower bound)."""
    T = ids.shape[1]
    out = model(input_ids=ids)
    logits = out.logits[:, T - K - 1: T - 1, :].float()
    tgt = ids[:, T - K: T]
    lp = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
    )
    return lp.mean() if reduction == "mean" else lp


def teacher_window_logits(model, ids, K):
    """Log-probs of the teacher on the last-K-token predictions (distillation target)."""
    T = ids.shape[1]
    out = model(input_ids=ids)
    return out.logits[0, T - K - 1: T - 1, :].float().log_softmax(-1)


def save_json(obj, name):
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", name)
    with open(out, "w") as f:
        json.dump(obj, f, indent=2)
    return out
