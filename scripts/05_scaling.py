"""Experiment 5 (H3) — does generalization improve with training-data scale?

Mirrors the Text-to-LoRA ablation (16 -> 479 tasks improved zero-shot). We hold a
fixed held-out set and grow the (nested) training set; for each size we train a
fresh hypernetwork and measure held-out gap-closed. If held-out climbs with N, the
held-out failure is a data-scale issue (the generator can't memorize), not an
architectural dead-end.

Full-batch gradient via chunked accumulation (deterministic -> stable, unlike the
stochastic minibatch that diverged).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, many_sequences, recent_window_loss_batch,
                    recent_window_logits_batch, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, prep_batch, apply_lora, clear_lora

T, K, R = 2048, 64, 16
TARGETS = ("q_proj", "v_proj")
STEPS, LR, CLIP, CHUNK = 400, 2e-4, 1.0, 32
N_HELD = 24
SIZES = [8, 16, 32, 64, 128, 256]
CONFIGS = [("narrativeqa", 0), ("gov_report", 0), ("multifieldqa_en", 0), ("qasper", 0),
           ("hotpotqa", 0), ("multi_news", 0), ("2wikimqa", 0), ("qmsum", 0), ("triviaqa", 0)]

tok, model = load_model()
d_model = model.config.hidden_size
pool = many_sequences(tok, CONFIGS, T, max_total=max(SIZES) + N_HELD + 20, per_doc=3)
g = torch.Generator().manual_seed(0)
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
held, train_pool = pool[:N_HELD], pool[N_HELD:]
print(f"pool={len(pool)}  held={len(held)}  train_pool={len(train_pool)}  sizes={SIZES}")


def prep(seqs, bs=8):
    ids = torch.cat(seqs, 0)
    lps, tces, Hs = [], [], []
    for s in range(0, ids.shape[0], bs):
        lp, ce, H = prep_batch(model, ids[s:s + bs], K)
        lps.append(lp); tces.append(ce); Hs.append(H)
    return ids, torch.cat(lps), torch.cat(tces), torch.cat(Hs)

he_ids, he_lp, he_tce, he_H = prep(held)
tp_ids, tp_lp, tp_tce, tp_H = prep(train_pool)


def gap_closed(hyper, wrapped, ids, H, tce):
    with torch.no_grad():
        apply_lora(wrapped, hyper(H)); adapted = recent_window_loss_batch(model, ids, K)
        clear_lora(wrapped); evict = recent_window_loss_batch(model, ids, K)
    t, e, a = tce.mean().item(), evict.mean().item(), adapted.mean().item()
    return (e - a) / (e - t) * 100


results = []
for N in SIZES:
    idx = list(range(N))
    wrapped = inject(model, targets=TARGETS, r=R)
    hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
    opt = torch.optim.Adam(hyper.parameters(), lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        for s in range(0, N, CHUNK):                      # full-batch via chunk accumulation
            ci = idx[s:s + CHUNK]
            apply_lora(wrapped, hyper(tp_H[ci]))
            stu = recent_window_logits_batch(model, tp_ids[ci], K)
            loss = F.kl_div(stu, tp_lp[ci], log_target=True, reduction="none").sum(-1).sum() / (N * K)
            loss.backward(); clear_lora(wrapped)
        torch.nn.utils.clip_grad_norm_(hyper.parameters(), CLIP); opt.step()
    hyper.eval()
    trc = gap_closed(hyper, wrapped, tp_ids[idx], tp_H[idx], tp_tce[idx])
    hec = gap_closed(hyper, wrapped, he_ids, he_H, he_tce)
    results.append(dict(n_train=N, train_closed=trc, held_closed=hec))
    print(f"N_train={N:>4}  train_closed={trc:6.1f}%  held_closed={hec:6.1f}%", flush=True)
    remove(model)
    save_json(dict(T=T, K=K, r=R, steps=STEPS, n_held=N_HELD, results=results), "05_scaling.json")  # incremental

print("\nsaved -> results/05_scaling.json")
