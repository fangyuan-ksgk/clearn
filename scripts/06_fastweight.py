"""Experiment 6 (H3 complement) — training-free fast-weight from the distant KV.

The founding intuition made literal: build a per-head linear-attention memory
S_h = Σ φ(k_j) ⊗ v_j from the distant KV (no training) and blend its read into each
recent query. Because S is computed from the sequence's own KV it is never OOD —
if it works it generalizes for free.

(a) global blend α swept (fully training-free)
(b) per-layer α (24 scalars) lightly tuned on train docs, evaluated on held-out
    (24 params can't memorize content -> a clean generalization test)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, many_sequences, recent_window_loss,
                    teacher_window_loss, teacher_window_logits, recent_window_logits, save_json)
from clearn import fastweight as fw

T, K = 2048, 64
CONFIGS = [("narrativeqa", 0), ("gov_report", 0), ("multifieldqa_en", 0), ("qasper", 0)]
tok, model = load_model()
n_layers = model.config.num_hidden_layers
pool = many_sequences(tok, CONFIGS, T, max_total=40, per_doc=2)
g = torch.Generator().manual_seed(0)
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
train, held = pool[:24], pool[24:32]
print(f"train={len(train)} held={len(held)}")


def baselines(docs):
    with torch.no_grad():
        t = np.mean([teacher_window_loss(model, x, K).item() for x in docs])
        e = np.mean([recent_window_loss(model, x, K).item() for x in docs])
    return float(t), float(e)

tr_t, tr_e = baselines(train)
he_t, he_e = baselines(held)
print(f"TRAIN teacher={tr_t:.4f} evict={tr_e:.4f} | HELD teacher={he_t:.4f} evict={he_e:.4f}\n")

fw.enable()


def encode(x):
    fw.FW.reset(); fw.FW.K = K; fw.FW.mode = "encode"
    with torch.no_grad():
        model(input_ids=x)
    return {k: v for k, v in fw.FW.store.items()}


def adapted_loss(x, store, scalar_or_vec):
    fw.FW.store = store; fw.FW.mode = "score"; fw.FW.lam = scalar_or_vec
    out = recent_window_loss(model, x, K)
    fw.FW.mode = "off"
    return out

# (a) training-free global alpha sweep
stores_tr = [encode(x) for x in train]
print("(a) training-free global alpha:")
best = (0.0, 0.0)
for a in [0.0, 0.02, 0.05, 0.1, 0.2]:
    with torch.no_grad():
        v = np.mean([adapted_loss(x, s, a).item() for x, s in zip(train, stores_tr)])
    closed = (tr_e - v) / (tr_e - tr_t) * 100
    if closed > best[1]:
        best = (a, closed)
    print(f"   alpha={a:>4}  train_closed={closed:6.1f}%")

# (b) per-layer alpha (sigmoid-gated), lightly tuned on train, eval held-out
fw.FW.mode = "off"
raw = torch.zeros(n_layers, device="cuda", requires_grad=True)   # sigmoid(0)=0.5 start -> opt down
opt = torch.optim.Adam([raw], lr=0.05)
with torch.no_grad():
    tr_lp = [teacher_window_logits(model, x, K) for x in train]
print("\n(b) per-layer alpha (24 gates) tuned on train:")
for step in range(120):
    opt.zero_grad(); tot = 0.0
    for x, s, lp in zip(train, stores_tr, tr_lp):
        fw.FW.store = s; fw.FW.mode = "score"; fw.FW.lam = torch.sigmoid(raw)
        stu = recent_window_logits(model, x, K)
        loss = F.kl_div(stu, lp, log_target=True, reduction="none").sum(-1).mean() / len(train)
        loss.backward(); tot += loss.item()
    fw.FW.mode = "off"
    opt.step()
    if step % 30 == 0 or step == 119:
        print(f"   step {step:>3}  KL={tot:.4f}")

a_vec = torch.sigmoid(raw).detach()
stores_he = [encode(x) for x in held]
with torch.no_grad():
    tr_a = np.mean([adapted_loss(x, s, a_vec).item() for x, s in zip(train, stores_tr)])
    he_a = np.mean([adapted_loss(x, s, a_vec).item() for x, s in zip(held, stores_he)])
fw.disable()
tr_closed = (tr_e - tr_a) / (tr_e - tr_t) * 100
he_closed = (he_e - he_a) / (he_e - he_t) * 100
print(f"\nper-layer alpha range [{a_vec.min():.3f},{a_vec.max():.3f}]")
print(f"TRAIN    closed={tr_closed:.1f}%   HELD-OUT closed={he_closed:.1f}%")
save_json(dict(T=T, K=K, train_freeform_best=best, n_layers=n_layers,
               perlayer=dict(train_closed=tr_closed, held_closed=he_closed,
                             alpha_min=float(a_vec.min()), alpha_max=float(a_vec.max()))),
          "06_fastweight.json")
print("saved -> results/06_fastweight.json")
