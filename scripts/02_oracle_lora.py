"""Experiment 2 (H1) — oracle per-sequence LoRA: can weights absorb distant KV?

For each of a few documents, directly optimize a LoRA to minimize the *evicted*
recent-window loss at K=64 (window-only forward — distant tokens never processed,
so the LoRA is the only channel for distant information). This is the upper bound
on what a weight delta can recover; the hypernetwork (Exp 3) must approach it
without test-time gradient descent.

Sweep rank and target modules to see gap-closed-per-parameter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, numpy as np
from clearn import (load_model, load_docs, tokenize_fixed,
                    recent_window_loss, teacher_window_loss, save_json)
from clearn.lora import inject, remove, alloc_oracle

T, K = 2048, 64
STEPS = 80
LR = 5e-3
CONFIGS = [("narrativeqa", 3), ("gov_report", 3)]
SETTINGS = [
    (("q_proj", "v_proj"), 8),
    (("q_proj", "v_proj"), 16),
    (("q_proj", "v_proj", "o_proj"), 16),
    (("q_proj", "k_proj", "v_proj", "o_proj"), 16),
]

tok, model = load_model()
docs = [d for cfg, n in CONFIGS for d in load_docs(cfg, n)]
ids_list = [x for d in docs if (x := tokenize_fixed(tok, d, T)) is not None]
print(f"{len(ids_list)} docs, T={T}, K={K}")

with torch.no_grad():
    teacher = float(np.mean([teacher_window_loss(model, x, K).item() for x in ids_list]))
    evict = float(np.mean([recent_window_loss(model, x, K).item() for x in ids_list]))
print(f"teacher={teacher:.4f}  evict={evict:.4f}  gap={evict-teacher:.4f}\n")

results = []
for targets, r in SETTINGS:
    per_doc, nparam = [], 0
    for x in ids_list:
        wrapped = inject(model, targets=targets, r=r)
        params = alloc_oracle(wrapped, device="cuda", dtype=torch.float32)
        nparam = sum(p.numel() for p in params)
        opt = torch.optim.Adam(params, lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            recent_window_loss(model, x, K).backward()
            opt.step()
        with torch.no_grad():
            per_doc.append(recent_window_loss(model, x, K).item())
        remove(model)
    adapted = float(np.mean(per_doc))
    closed = (evict - adapted) / (evict - teacher) * 100
    results.append(dict(targets=list(targets), r=r, params=nparam, adapted=adapted, pct_gap_closed=closed))
    print(f"{'+'.join(targets):<30} r={r:<3} {nparam/1e3:>6.0f}k params  adapted={adapted:.4f}  gap_closed={closed:>5.1f}%")

save_json(dict(T=T, K=K, steps=STEPS, lr=LR, teacher=teacher, evict=evict, results=results),
          "02_oracle_lora.json")
print("\nsaved -> results/02_oracle_lora.json")
