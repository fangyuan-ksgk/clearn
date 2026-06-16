"""Experiment 8 (H3) — meta-train the hypernet as a LAUNCH POINT for test-time steps.

Exp 7 puzzle: the feed-forward hypernet init *loses* to a cold start once K≥8,
because it is trained to be the best *zero-step* answer, not the best *initialization
for K gradient steps*. Fix (FOMAML / Reptile, cf. Tancik 2012.02189): meta-train the
hypernet THROUGH a few inner test-time steps so its output adapts well.

We train two hypernets on the SAME data split and compare their held-out
test-time-adaptation curves: feed-forward-trained vs FOMAML-trained.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, many_sequences, recent_window_loss_batch,
                    recent_window_logits_batch, recent_window_logits,
                    teacher_window_loss, teacher_window_logits, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, prep_batch, apply_lora, clear_lora

T, K, R = 2048, 64, 16
TARGETS = ("q_proj", "v_proj")
N_TRAIN, N_HELD = 16, 8
INNER_K, INNER_LR = 3, 3e-3          # meta inner loop
OUTER_STEPS, OUTER_LR = 100, 3e-4
FF_STEPS = 300                       # feed-forward baseline training
KS = [0, 2, 4, 8, 16]
CONFIGS = [("narrativeqa", 0), ("gov_report", 0), ("multifieldqa_en", 0), ("qasper", 0), ("hotpotqa", 0)]

tok, model = load_model()
d_model = model.config.hidden_size
pool = many_sequences(tok, CONFIGS, T, max_total=N_TRAIN + N_HELD + 10, per_doc=2)
g = torch.Generator().manual_seed(1)                  # SAME split as Exp 7
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
train, held = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_HELD]
print(f"train={len(train)} held={len(held)}  inner_K={INNER_K} inner_lr={INNER_LR}")

tr_ids = torch.cat(train, 0)
with torch.no_grad():
    tr_lp, _, tr_H = prep_batch(model, tr_ids, K)
    tr_lp_list = [teacher_window_logits(model, x, K) for x in train]
wrapped = inject(model, targets=TARGETS, r=R)
M = len(wrapped)


def he_baselines():
    with torch.no_grad():
        ids = torch.cat(held, 0)
        t = np.mean([teacher_window_loss(model, x, K).item() for x in held])
        e = recent_window_loss_batch(model, ids, K).mean().item()
    return float(t), float(e)

he_t, he_e = he_baselines()
print(f"HELD-OUT teacher={he_t:.4f} evict={he_e:.4f}\n")


def distill(x, factors, lp):
    apply_lora(wrapped, factors)
    return F.kl_div(recent_window_logits(model, x, K), lp, log_target=True, reduction="none").sum(-1).mean()


def ttt_curve(hyper):
    """held-out gap-closed vs #test-time steps (hyper init, full ΔW)."""
    hyper.eval(); out = {}
    for nsteps in KS:
        losses = []
        for x in held:
            with torch.no_grad():
                _, _, H = prep_batch(model, x, K)
                init = hyper(H); lp = teacher_window_logits(model, x, K)
            params = [t.detach().clone().requires_grad_() for AB in init for t in AB]
            fac = [(params[2 * i], params[2 * i + 1]) for i in range(M)]
            if nsteps > 0:
                o = torch.optim.Adam(params, lr=INNER_LR)
                for _ in range(nsteps):
                    o.zero_grad(); distill(x, fac, lp).backward(); o.step()
                clear_lora(wrapped)
            with torch.no_grad():
                apply_lora(wrapped, fac); losses.append(recent_window_loss_batch(model, x, K).item()); clear_lora(wrapped)
        out[nsteps] = (he_e - np.mean(losses)) / (he_e - he_t) * 100
    return out


# ---- (A) feed-forward-trained hypernet (reuse saved curve if present) ----
_prev = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "08_meta.json")
ff_curve = None
if os.path.exists(_prev):
    pj = json.load(open(_prev))
    if pj.get("feedforward"):
        ff_curve = {int(k): v for k, v in pj["feedforward"].items()}
        print("reusing saved feed-forward curve:", {k: round(v, 1) for k, v in ff_curve.items()})
if ff_curve is None:
    print("training feed-forward hypernet...")
    hf = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
    opt = torch.optim.Adam(hf.parameters(), lr=3e-4)
    for s in range(FF_STEPS):
        apply_lora(wrapped, hf(tr_H))
        loss = F.kl_div(recent_window_logits_batch(model, tr_ids, K), tr_lp, log_target=True, reduction="none").sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(hf.parameters(), 1.0); opt.step(); clear_lora(wrapped)
    ff_curve = ttt_curve(hf)
    print("feed-forward init TTT curve:", {k: round(v, 1) for k, v in ff_curve.items()}, flush=True)
    save_json(dict(T=T, K=K, n_train=N_TRAIN, n_held=N_HELD, inner_k=INNER_K,
                   held_teacher=he_t, held_evict=he_e, feedforward=ff_curve, fomaml=None), "08_meta.json")

# ---- (B) FOMAML-trained hypernet (meta launch point) ----
print("meta-training (FOMAML) hypernet...")
hm = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
mopt = torch.optim.Adam(hm.parameters(), lr=OUTER_LR)
for s in range(OUTER_STEPS):
    mopt.zero_grad(); tot = 0.0
    for x, lp in zip(train, tr_lp_list):
        with torch.no_grad():
            _, _, H = prep_batch(model, x, K)
        f0 = hm(H)                                            # carries grad to hm
        inner = [(A.detach().clone().requires_grad_(), B.detach().clone().requires_grad_()) for (A, B) in f0]
        io = torch.optim.SGD([t for ab in inner for t in ab], lr=INNER_LR)
        for _ in range(INNER_K):                              # inner adaptation (detached from hm)
            io.zero_grad(); distill(x, inner, lp).backward(); io.step(); clear_lora(wrapped)
        # FOMAML re-anchor: value = adapted, grad flows to hm via f0
        fac = [(f0[i][0] + (inner[i][0] - f0[i][0]).detach(),
                f0[i][1] + (inner[i][1] - f0[i][1]).detach()) for i in range(M)]
        oloss = distill(x, fac, lp) / len(train)
        oloss.backward(); clear_lora(wrapped); tot += oloss.item()
    torch.nn.utils.clip_grad_norm_(hm.parameters(), 1.0); mopt.step()
    if s % 40 == 0 or s == OUTER_STEPS - 1:
        print(f"  outer {s:>3}  post-adapt KL={tot:.4f}")
meta_curve = ttt_curve(hm)

print(f"\n{'K':>4} {'feedfwd-init':>13} {'FOMAML-init':>12}")
for k in KS:
    print(f"{k:>4} {ff_curve[k]:>12.1f}% {meta_curve[k]:>11.1f}%")
remove(model)
save_json(dict(T=T, K=K, n_train=N_TRAIN, n_held=N_HELD, inner_k=INNER_K,
               held_teacher=he_t, held_evict=he_e,
               feedforward=ff_curve, fomaml=meta_curve), "08_meta.json")
print("\nsaved -> results/08_meta.json")
