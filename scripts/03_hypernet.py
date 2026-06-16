"""Experiment 3 (H2) — the hypernetwork.

One shared hypernetwork reads each document's *distant* hidden states and emits a
LoRA in a single forward pass. Trained by context distillation: the adapted
student (window-only forward — distant KV gone — + generated LoRA) is pushed to
match the full-KV teacher's next-token distribution on the recent window. Because
the target is the teacher, the student is pulled toward (not past) teacher quality
— avoiding the memorization pathology seen with the oracle.

Compare: teacher (full KV) / evicted (no adapter) / hypernet-adapted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, load_docs, tokenize_fixed, recent_window_loss,
                    recent_window_logits, teacher_window_loss, teacher_window_logits, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, encode_distant, apply_lora, clear_lora

T, K = 2048, 64
R = 16
TARGETS = ("q_proj", "v_proj")
STEPS = 400
LR = 3e-4
CLIP = 1.0
ENCODE_LAYER = -1
CONFIGS = [("narrativeqa", 4), ("gov_report", 4)]

tok, model = load_model()
d_model = model.config.hidden_size
docs = [d for cfg, n in CONFIGS for d in load_docs(cfg, n)]
ids_list = [x for d in docs if (x := tokenize_fixed(tok, d, T)) is not None]
print(f"{len(ids_list)} docs, T={T}, K={K}, rank={R}, targets={TARGETS}")

# teacher distillation targets + baselines (all on the same last-K predictions)
with torch.no_grad():
    teacher_lp = [teacher_window_logits(model, x, K) for x in ids_list]
    teacher_ce = [teacher_window_loss(model, x, K).item() for x in ids_list]
    evict_ce = [recent_window_loss(model, x, K).item() for x in ids_list]
    H_list = [encode_distant(model, x, K, layer=ENCODE_LAYER) for x in ids_list]
teacher = float(np.mean(teacher_ce)); evict = float(np.mean(evict_ce))
print(f"teacher_CE={teacher:.4f}  evict_CE={evict:.4f}  gap={evict-teacher:.4f}\n")

wrapped = inject(model, targets=TARGETS, r=R)
hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
nparam = sum(p.numel() for p in hyper.parameters())
print(f"hypernet params: {nparam/1e6:.2f}M  (writes LoRA to {len(wrapped)} matrices)\n")
opt = torch.optim.Adam(hyper.parameters(), lr=LR)

for step in range(STEPS):
    opt.zero_grad()
    total = 0.0
    for i in range(len(ids_list)):          # full-batch: average over all docs (stable)
        apply_lora(wrapped, hyper(H_list[i]))
        stu = recent_window_logits(model, ids_list[i], K)
        loss = F.kl_div(stu, teacher_lp[i], log_target=True, reduction="batchmean") / len(ids_list)
        loss.backward()
        clear_lora(wrapped)
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(hyper.parameters(), CLIP)
    opt.step()
    if step % 25 == 0 or step == STEPS - 1:
        print(f"step {step:>4}  KL={total:.4f}")

# eval on trained docs
hyper.eval()
adapted_ce = []
with torch.no_grad():
    for x, H in zip(ids_list, H_list):
        apply_lora(wrapped, hyper(H))
        adapted_ce.append(recent_window_loss(model, x, K).item())
        clear_lora(wrapped)
adapted = float(np.mean(adapted_ce))
closed = (evict - adapted) / (evict - teacher) * 100
print(f"\nteacher_CE={teacher:.4f}  evict_CE={evict:.4f}  adapted_CE={adapted:.4f}")
print(f"gap closed: {closed:.1f}%   PPL {np.exp(evict):.1f} -> {np.exp(adapted):.1f} (teacher {np.exp(teacher):.1f})")

remove(model)
save_json(dict(T=T, K=K, r=R, targets=list(TARGETS), steps=STEPS, lr=LR,
               hyper_params=nparam, teacher=teacher, evict=evict, adapted=adapted,
               pct_gap_closed=closed, per_doc_adapted=adapted_ce,
               per_doc_teacher=teacher_ce, per_doc_evict=evict_ce), "03_hypernet.json")
print("saved -> results/03_hypernet.json")
