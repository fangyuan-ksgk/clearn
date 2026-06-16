# clearn — online KV-cache compression via weight adaptation

Train a **hypernetwork** that reads the *distant* part of a sequence's context
and emits a **LoRA** weight delta, so the model can **evict the distant KV cache**
while keeping next-token quality. It converts memory-stored-as-KV into
memory-stored-as-weights — an online KV-cache compression machine.

See **[NOTES.md](NOTES.md)** for the running narrative, findings, research
grounding (Doc-to-LoRA, Generative Adapter, Fast Weight Programmers, context
distillation), and plan.

## Layout
- `clearn/core.py` — model/data loading, the eviction attention-mask, window-loss metric
- `clearn/lora.py` — manual LoRA injection (factors can be learned or generated)
- `clearn/hypernet.py` — Perceiver-style hypernetwork: distant hidden states → LoRA
- `scripts/01_baseline_curve.py` — the cost of evicting distant KV (the problem)
- `scripts/02_oracle_lora.py` — per-sequence oracle LoRA (capacity upper bound)
- `scripts/03_hypernet.py` — the hypernetwork, trained by context distillation
- `results/` — saved JSON metrics

Model: Qwen2.5-0.5B (open). Data: LongBench (`data/data/*.jsonl`).
