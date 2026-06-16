"""Experiment 3b (H3) — does the hypernetwork generalise, or just memorise?

Train the shared hypernetwork on TRAIN docs, then evaluate it *frozen* on HELD-OUT
docs it never saw. High held-out gap-closed => genuine online KV->weight
compression (Doc-to-LoRA claim). Only-train => per-doc memorization.

Batched per-sample LoRA: every doc in a minibatch gets its own generated LoRA and
all are scored in a single model forward (~30x faster than the per-doc loop).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, load_docs, tokenize_fixed, recent_window_loss_batch,
                    recent_window_logits_batch, teacher_window_loss, teacher_window_logits, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, encode_distant, apply_lora, clear_lora

T, K, R = 2048, 64, 16
TARGETS = ("q_proj", "v_proj")
STEPS, LR, CLIP = 800, 2e-4, 1.0   # full-batch over all train docs (cheap with batching) = stable
CONFIGS = [("narrativeqa", 8), ("gov_report", 8), ("multifieldqa_en", 8),
           ("qasper", 6), ("hotpotqa", 6), ("multi_news", 6)]

tok, model = load_model()
d_model = model.config.hidden_size
docs = [d for cfg, n in CONFIGS for d in load_docs(cfg, n)]
ids_all = [x for d in docs if (x := tokenize_fixed(tok, d, T)) is not None]
n_held = max(4, len(ids_all) // 5)
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(ids_all), generator=g).tolist()
ids_all = [ids_all[i] for i in perm]                  # shuffle so held-out mixes configs
tr, he = ids_all[:-n_held], ids_all[-n_held:]
print(f"{len(tr)} train docs, {len(he)} held-out, T={T}, K={K}, rank={R} (full-batch)")


def prep(ids_list):
    ids = torch.cat(ids_list, 0)                       # [n,T]
    with torch.no_grad():
        lp = torch.stack([teacher_window_logits(model, x, K) for x in ids_list])   # [n,K,V]
        tce = torch.tensor([teacher_window_loss(model, x, K).item() for x in ids_list])
        H = torch.stack([encode_distant(model, x, K) for x in ids_list])           # [n,M,d]
    return ids, lp, tce, H

tr_ids, tr_lp, tr_tce, tr_H = prep(tr)
he_ids, he_lp, he_tce, he_H = prep(he)

wrapped = inject(model, targets=TARGETS, r=R)
hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
print(f"hypernet params: {sum(p.numel() for p in hyper.parameters())/1e6:.2f}M\n")
opt = torch.optim.Adam(hyper.parameters(), lr=LR)


def gap_closed(ids, H, tce):
    with torch.no_grad():
        apply_lora(wrapped, hyper(H))
        adapted = recent_window_loss_batch(model, ids, K)
        clear_lora(wrapped)
        evict = recent_window_loss_batch(model, ids, K)     # LoRA cleared -> baseline
    t, e, a = tce.mean().item(), evict.mean().item(), adapted.mean().item()
    return t, e, a, (e - a) / (e - t) * 100

for step in range(STEPS):
    apply_lora(wrapped, hyper(tr_H))                  # full batch: all train docs
    stu = recent_window_logits_batch(model, tr_ids, K)
    loss = F.kl_div(stu, tr_lp, log_target=True, reduction="none").sum(-1).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(hyper.parameters(), CLIP); opt.step()
    clear_lora(wrapped)
    if step % 50 == 0 or step == STEPS - 1:
        hyper.eval()
        _, _, _, trc = gap_closed(tr_ids, tr_H, tr_tce)
        _, _, _, hec = gap_closed(he_ids, he_H, he_tce)
        hyper.train()
        print(f"step {step:>4}  KL={loss.item():.4f}  train_closed={trc:6.1f}%  held_closed={hec:6.1f}%")

hyper.eval()
tr_r = gap_closed(tr_ids, tr_H, tr_tce)
he_r = gap_closed(he_ids, he_H, he_tce)
print(f"\nTRAIN    teacher={tr_r[0]:.4f} evict={tr_r[1]:.4f} adapted={tr_r[2]:.4f}  closed={tr_r[3]:.1f}%")
print(f"HELD-OUT teacher={he_r[0]:.4f} evict={he_r[1]:.4f} adapted={he_r[2]:.4f}  closed={he_r[3]:.1f}%")
remove(model)
save_json(dict(T=T, K=K, r=R, targets=list(TARGETS), steps=STEPS,
               n_train=len(tr), n_held=len(he),
               train=dict(teacher=tr_r[0], evict=tr_r[1], adapted=tr_r[2], closed=tr_r[3]),
               held=dict(teacher=he_r[0], evict=he_r[1], adapted=he_r[2], closed=he_r[3])),
          "04_generalization.json")
print("saved -> results/04_generalization.json")
