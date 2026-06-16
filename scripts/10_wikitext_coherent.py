"""Experiment 10 — test the structure hypothesis from Exp 9.

Exp 9: amortized generalization failed on WikiText windows that were *concatenated
across unrelated articles* (heterogeneous tasks). Meta-learning theory says
generalization needs shared structure across tasks. Prediction: use COHERENT
single-article windows (more homogeneous) and held-out generalization should
partially return. Same pipeline as Exp 9; only the pool changes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from datasets import load_dataset
from clearn import (load_model, recent_window_loss_batch, recent_window_logits_batch,
                    teacher_window_loss, recent_window_loss, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, prep_batch, apply_lora, clear_lora

T, K, R = 2048, 64, 16
TARGETS = ("q_proj", "v_proj")
STEPS, LR, CLIP, CHUNK = 400, 2e-4, 1.0, 32
N_HELD = 24
SIZES = [8, 16, 32, 64, 128]

tok, model = load_model()
d_model = model.config.hidden_size


def is_title(t):
    s = t.strip()
    return s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")


def coherent_articles(n_seqs, T):
    """One T-token window from the START of each COHERENT single article (>= T tokens)."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    seqs, cur = [], []
    for ex in ds:
        t = ex["text"]
        if is_title(t):
            if cur:
                ids = tok(" ".join(cur), add_special_tokens=False).input_ids
                if len(ids) >= T:
                    seqs.append(torch.tensor(ids[:T], device="cuda")[None])
                    if len(seqs) >= n_seqs:
                        return seqs
            cur = [t]
        elif len(t.strip()) > 0:
            cur.append(t)
    return seqs

pool = coherent_articles(max(SIZES) + N_HELD + 20, T)
g = torch.Generator().manual_seed(0)
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
held, train_pool = pool[:N_HELD], pool[N_HELD:]
print(f"WikiText coherent articles: pool={len(pool)} held={len(held)} train_pool={len(train_pool)}")

with torch.no_grad():
    teach = np.mean([teacher_window_loss(model, x, K).item() for x in held])
    evic = np.mean([recent_window_loss(model, x, K).item() for x in held])
print(f"[eviction] teacher={teach:.4f} evict={evic:.4f} gap={evic-teach:.4f}\n")


def prep(seqs):
    ids = torch.cat(seqs, 0); lps, tces, Hs = [], [], []
    for s in range(0, ids.shape[0], 8):
        lp, ce, H = prep_batch(model, ids[s:s + 8], K)
        lps.append(lp); tces.append(ce); Hs.append(H)
    return ids, torch.cat(lps), torch.cat(tces), torch.cat(Hs)

he_ids, he_lp, he_tce, he_H = prep(held)
tp_ids, tp_lp, tp_tce, tp_H = prep(train_pool)


def gap_closed(hyper, ids, H, tce):
    with torch.no_grad():
        apply_lora(wrapped, hyper(H)); a = recent_window_loss_batch(model, ids, K)
        clear_lora(wrapped); e = recent_window_loss_batch(model, ids, K)
    t, e, a = tce.mean().item(), e.mean().item(), a.mean().item()
    return (e - a) / (e - t) * 100

results = []
print(f"[scaling]  {'N':>6} {'train':>8} {'held-out':>9}")
for N in SIZES:
    idx = list(range(N))
    wrapped = inject(model, targets=TARGETS, r=R)
    hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
    opt = torch.optim.Adam(hyper.parameters(), lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        for s in range(0, N, CHUNK):
            ci = idx[s:s + CHUNK]
            apply_lora(wrapped, hyper(tp_H[ci]))
            loss = F.kl_div(recent_window_logits_batch(model, tp_ids[ci], K), tp_lp[ci],
                            log_target=True, reduction="none").sum(-1).sum() / (N * K)
            loss.backward(); clear_lora(wrapped)
        torch.nn.utils.clip_grad_norm_(hyper.parameters(), CLIP); opt.step()
    hyper.eval()
    trc = gap_closed(hyper, tp_ids[idx], tp_H[idx], tp_tce[idx])
    hec = gap_closed(hyper, he_ids, he_H, he_tce)
    results.append(dict(n_train=N, train_closed=trc, held_closed=hec))
    print(f"           {N:>6} {trc:>7.1f}% {hec:>8.1f}%", flush=True)
    remove(model)
    save_json(dict(dataset="wikitext-103-coherent", T=T, K=K, teacher=teach, evict=evic,
                   n_held=N_HELD, results=results), "10_wikitext_coherent.json")

print("\nsaved -> results/10_wikitext_coherent.json")
