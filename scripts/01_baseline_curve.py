"""Experiment 1 — the cost of evicting distant KV.

For a handful of real LongBench documents, sweep the recent-window size K and
measure next-token loss on the recent window under: full KV (teacher), local
window only (evict), and local window + 4-token attention sink.

This quantifies the gap the hypernetwork must close.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from clearn import load_model, load_docs, tokenize_fixed, window_loss, save_json

T = 2048
KS = [1024, 512, 256, 128, 64]
CONFIGS = [("narrativeqa", 4), ("gov_report", 4)]

tok, model = load_model()
docs = [d for cfg, n in CONFIGS for d in load_docs(cfg, n)]
ids_list = [x for d in docs if (x := tokenize_fixed(tok, d, T)) is not None]
print(f"{len(ids_list)} docs at T={T}")

rows = []
header = f"{'K':>6} {'%KVkept':>8} {'teacher':>9} {'evict':>9} {'evict+sink4':>12} {'gap':>7}"
print(header)
import torch
for K in KS:
  with torch.no_grad():
    tl = [window_loss(model, x, K, evict=False) for x in ids_list]
    el = [window_loss(model, x, K, evict=True, sink=0) for x in ids_list]
    es = [window_loss(model, x, K, evict=True, sink=4) for x in ids_list]
    t, e, s = np.mean(tl), np.mean(el), np.mean(es)
    rows.append(dict(K=K, kv_kept=K / T, teacher=t, evict=e, evict_sink4=s, gap=e - t))
    print(f"{K:>6} {100*K/T:>7.0f}% {t:>9.4f} {e:>9.4f} {s:>12.4f} {e-t:>7.4f}")

save_json(dict(model=model.config._name_or_path, T=T, n_docs=len(ids_list), rows=rows),
          "01_baseline_curve.json")
print("\nsaved -> results/01_baseline_curve.json")
