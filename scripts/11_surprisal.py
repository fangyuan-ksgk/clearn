"""Experiment 11 — what does the amortized recovery actually transfer: style or facts?

Operationalization on real held-out LongBench data (no needle construction, no model
switch): for each recent-window token, eviction-cost = loss(evict) - loss(teacher)
measures how much that token *needed the distant context*. High-cost tokens are the
content/"fact-like" ones; low-cost tokens are locally predictable (style/syntax).
We train the hypernet (strong-generalization regime, N=128) and ask, on HELD-OUT docs,
what fraction of each cost-tercile's gap it recovers. Style-only => recovery
concentrated in low-cost tokens; fact transfer => also recovers high-cost tokens.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn.functional as F, numpy as np
from clearn import (load_model, many_sequences, recent_window_loss, teacher_window_loss,
                    recent_window_logits_batch, save_json)
from clearn.lora import inject, remove
from clearn.hypernet import HyperLoRA, prep_batch, apply_lora, clear_lora

T, K, R, N_TRAIN, N_HELD = 2048, 64, 16, 128, 24
TARGETS = ("q_proj", "v_proj")
STEPS, LR, CLIP, CHUNK = 400, 2e-4, 1.0, 32
CONFIGS = [("narrativeqa", 0), ("gov_report", 0), ("multifieldqa_en", 0), ("qasper", 0),
           ("hotpotqa", 0), ("multi_news", 0), ("2wikimqa", 0), ("qmsum", 0)]

tok, model = load_model()
d_model = model.config.hidden_size
pool = many_sequences(tok, CONFIGS, T, max_total=N_TRAIN + N_HELD + 20, per_doc=3)
g = torch.Generator().manual_seed(0)
pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
held, train = pool[:N_HELD], pool[N_HELD:N_HELD + N_TRAIN]
print(f"train={len(train)} held={len(held)}")

tr_ids = torch.cat(train, 0)
with torch.no_grad():                                   # chunked to avoid OOM (shared GPU)
    lps, Hs = [], []
    for s in range(0, tr_ids.shape[0], 8):
        lp, _, H = prep_batch(model, tr_ids[s:s + 8], K)
        lps.append(lp); Hs.append(H)
    tr_lp, tr_H = torch.cat(lps), torch.cat(Hs)

wrapped = inject(model, targets=TARGETS, r=R)
hyper = HyperLoRA(wrapped, d_model=d_model, r=R).cuda()
opt = torch.optim.Adam(hyper.parameters(), lr=LR)
print("training hypernet (N=128)...")
idx = list(range(N_TRAIN))
for step in range(STEPS):
    opt.zero_grad()
    for s in range(0, N_TRAIN, CHUNK):
        ci = idx[s:s + CHUNK]
        apply_lora(wrapped, hyper(tr_H[ci]))
        loss = F.kl_div(recent_window_logits_batch(model, tr_ids[ci], K), tr_lp[ci],
                        log_target=True, reduction="none").sum(-1).sum() / (N_TRAIN * K)
        loss.backward(); clear_lora(wrapped)
    torch.nn.utils.clip_grad_norm_(hyper.parameters(), CLIP); opt.step()
hyper.eval()

# per-token decomposition on held-out
costs, recs = [], []
with torch.no_grad():
    for x in held:
        t = teacher_window_loss(model, x, K, reduction="none")
        e = recent_window_loss(model, x, K, reduction="none")
        _, _, H = prep_batch(model, x, K)
        apply_lora(wrapped, hyper(H)); a = recent_window_loss(model, x, K, reduction="none"); clear_lora(wrapped)
        costs.append((e - t)); recs.append((e - a))
cost = torch.cat(costs); rec = torch.cat(recs)
overall = rec.sum().item() / cost.sum().item() * 100
print(f"\nHELD-OUT overall gap closed: {overall:.1f}%  (n_tokens={cost.numel()})")

# terciles by eviction cost
q1, q2 = torch.quantile(cost, torch.tensor([1/3, 2/3], device=cost.device))
bins = [("low-cost (style)", cost <= q1), ("mid", (cost > q1) & (cost <= q2)), ("high-cost (facts)", cost > q2)]
print(f"\n{'token bin':>20} {'n':>6} {'mean cost':>10} {'recovered%':>11}")
rows = []
for name, m in bins:
    c, r = cost[m], rec[m]
    frac = r.sum().item() / c.sum().item() * 100 if c.sum() > 0 else 0
    rows.append(dict(bin=name, n=int(m.sum()), mean_cost=c.mean().item(), recovered_pct=frac))
    print(f"{name:>20} {int(m.sum()):>6} {c.mean().item():>10.3f} {frac:>10.1f}%")

# correlation: does recovery track cost?
corr = torch.corrcoef(torch.stack([cost, rec]))[0, 1].item()
print(f"\ncorr(eviction-cost, recovery) = {corr:.3f}")
remove(model)
save_json(dict(T=T, K=K, n_train=N_TRAIN, n_held=N_HELD, overall_closed=overall,
               bins=rows, corr_cost_recovery=corr), "11_surprisal.json")
print("saved -> results/11_surprisal.json")
