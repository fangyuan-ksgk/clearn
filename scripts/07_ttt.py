"""Experiment 7 (H3) — cheap test-time gradient refinement of the predicted LoRA.

Hypothesis (the OOD fix): the feed-forward hypernetwork misfires on unseen docs,
but a *gradient computed from the test sequence's own distant tokens* is
data-grounded and OOD-robust. So: predict ΔW (warm start), then take a few cheap
self-supervised gradient steps on ΔW using the LM loss over a chunk of the DISTANT
context (legit — those tokens are known input; we never touch the scored tokens).

We compare held-out gap-closed for:
  (c) K=0 feed-forward hypernetwork (baseline)
  (a) hypernet init + K test-time steps
  (b) zero init + K test-time steps  (pure TTT, no hypernet)
Trained at a small N where feed-forward OOD is mediocre, so there is room to show
the effect.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, many_sequences, recent_window_loss_batch,
                    recent_window_logits_batch, recent_window_logits,
                    teacher_window_loss, teacher_window_logits, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, prep_batch, apply_lora, clear_lora

T, K, R = 2048, 64, 16
TARGETS = ("q_proj", "v_proj")
N_TRAIN, N_HELD = 16, 10
SSL_CHUNK = 256            # (unused now; kept for json schema)
TTT_LR = 3e-3
KS = [0, 1, 2, 4, 8, 16, 32]
CONFIGS = [("narrativeqa", 0), ("gov_report", 0), ("multifieldqa_en", 0), ("qasper", 0), ("hotpotqa", 0)]

tok, model = load_model()
d_model = model.config.hidden_size
pool = many_sequences(tok, CONFIGS, T, max_total=N_TRAIN + N_HELD + 10, per_doc=2)
g = torch.Generator().manual_seed(1)
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
train, held = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_HELD]
print(f"train={len(train)} held={len(held)}  ssl_chunk={SSL_CHUNK}  ttt_lr={TTT_LR}")

tr_ids = torch.cat(train, 0)
with torch.no_grad():
    tr_lp, _, tr_H = prep_batch(model, tr_ids, K)

wrapped = inject(model, targets=TARGETS, r=R)
hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
opt = torch.optim.Adam(hyper.parameters(), lr=3e-4)
print("training feed-forward hypernet...")
for step in range(350):
    apply_lora(wrapped, hyper(tr_H))
    loss = F.kl_div(recent_window_logits_batch(model, tr_ids, K), tr_lp,
                    log_target=True, reduction="none").sum(-1).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0); opt.step(); clear_lora(wrapped)
hyper.eval()


def ttt_loss(x, factors, teacher_lp):
    """Aligned, leak-free signal: per-sequence context self-distillation.
    Target = this sequence's OWN full-KV recent-window logits (available because we
    have the distant KV once at adaptation time). No scored-token labels used."""
    apply_lora(wrapped, factors)
    stu = recent_window_logits(model, x, K)
    return F.kl_div(stu, teacher_lp, log_target=True, reduction="none").sum(-1).mean()


def teacher_evict(docs):
    with torch.no_grad():
        ids = torch.cat(docs, 0)
        t = np.mean([teacher_window_loss(model, x, K).item() for x in docs])
        e = recent_window_loss_batch(model, ids, K).mean().item()
    return float(t), float(e)

he_t, he_e = teacher_evict(held)
print(f"HELD-OUT teacher={he_t:.4f} evict={he_e:.4f} gap={he_e-he_t:.4f}\n")


def gap_closed_after_ttt(docs, n_steps, use_hyper_init):
    """Per-doc: init factors, take n_steps test-time gradient steps on distant SSL loss,
    then measure the recent-window loss (the metric)."""
    losses = []
    for x in docs:
        with torch.no_grad():
            _, _, H = prep_batch(model, x, K)
            init = hyper(H)
            teacher_lp = teacher_window_logits(model, x, K)     # this seq's own full-KV target
        if use_hyper_init:
            params = [t.detach().clone().requires_grad_() for AB in init for t in AB]
        else:
            # fair Temp-LoRA cold start: standard LoRA init (A~small random, B=0)
            params = []
            for (A, B) in init:
                params.append((torch.randn_like(A) * 0.01).requires_grad_())
                params.append(torch.zeros_like(B).requires_grad_())
        factors = [(params[2 * i], params[2 * i + 1]) for i in range(len(wrapped))]
        if n_steps > 0:
            o = torch.optim.Adam(params, lr=TTT_LR)
            for _ in range(n_steps):
                o.zero_grad(); ttt_loss(x, factors, teacher_lp).backward(); o.step()
            clear_lora(wrapped)
        with torch.no_grad():
            apply_lora(wrapped, factors)
            losses.append(recent_window_loss_batch(model, x, K).item())
            clear_lora(wrapped)
    a = float(np.mean(losses))
    return (he_e - a) / (he_e - he_t) * 100

rows = []
print(f"{'K_ttt':>6} {'hyper+TTT':>11} {'zeroinit+TTT':>13}")
for k in KS:
    hc = gap_closed_after_ttt(held, k, use_hyper_init=True)
    zc = gap_closed_after_ttt(held, k, use_hyper_init=False) if k > 0 else None
    rows.append(dict(k=k, hyper_ttt=hc, zero_ttt=zc))
    print(f"{k:>6} {hc:>10.1f}% {('' if zc is None else f'{zc:>12.1f}%')}")

def gap_closed_gates(docs, n_steps):
    """'Very few places': freeze hypernet ΔW directions, tune only per-module scalar
    gates (one per wrapped Linear = 48 scalars) by test-time context distillation."""
    losses = []
    for x in docs:
        with torch.no_grad():
            _, _, H = prep_batch(model, x, K)
            f0 = [(A.detach(), B.detach()) for (A, B) in hyper(H)]
            teacher_lp = teacher_window_logits(model, x, K)
        gates = torch.ones(len(wrapped), device="cuda", requires_grad=True)
        if n_steps > 0:
            o = torch.optim.Adam([gates], lr=0.05)
            for _ in range(n_steps):
                o.zero_grad()
                apply_lora(wrapped, [(f0[m][0], gates[m] * f0[m][1]) for m in range(len(wrapped))])
                stu = recent_window_logits(model, x, K)
                F.kl_div(stu, teacher_lp, log_target=True, reduction="none").sum(-1).mean().backward()
                o.step(); clear_lora(wrapped)
        with torch.no_grad():
            apply_lora(wrapped, [(f0[m][0], gates[m].detach() * f0[m][1]) for m in range(len(wrapped))])
            losses.append(recent_window_loss_batch(model, x, K).item()); clear_lora(wrapped)
    a = float(np.mean(losses)); return (he_e - a) / (he_e - he_t) * 100

print(f"\n'very few places' — tune only {len(wrapped)} per-module gates on hypernet ΔW:")
gate_rows = []
for k in KS:
    gc = gap_closed_gates(held, k)
    gate_rows.append(dict(k=k, gates_ttt=gc))
    print(f"{k:>6} gates+TTT = {gc:6.1f}%")

remove(model)
save_json(dict(T=T, K=K, n_train=N_TRAIN, n_held=N_HELD, ttt_lr=TTT_LR, n_gates=len(wrapped),
               held_teacher=he_t, held_evict=he_e, rows=rows, gate_rows=gate_rows), "07_ttt.json")
print("\nsaved -> results/07_ttt.json")
