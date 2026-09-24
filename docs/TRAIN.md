# TRAIN — recipe + reasoning

Config of record: `configs/{base,train,lora,data}.yaml`. Reasoning for the big choices is in
DECISIONS.md (cited inline); this doc is the operational recipe.

## Where it runs

- **Server (bf16, full GPU) — the training home.** Set `full_gpu: true` in `configs/base.yaml`;
  the base loads entirely on one GPU. `python scripts/train.py --smoke` then `python scripts/train.py`.
- **Laptop (RTX 3050 6 GB) — inference/eval only.** bf16 + CPU-offload training exhausts the
  11 GB RAM (OOM), so training is not run here (see RECON §6 / the Phase-4 note). A 4-bit QLoRA
  path is the only laptop-viable option and is a smoke-only fallback.

## Recipe

| Component | Setting | Why |
|---|---|---|
| Base | `LlamaForCausalLM` bf16, `attn_implementation=sdpa` | matches `reference/inference.py` |
| Adapter | LoRA r=32 α=64 dropout=0.05 | Decision 2 |
| LoRA targets | q_proj, v_proj (all 28 layers); up_proj, gate_proj (layers 8–24) | Decision 2 / `model/lora.py` |
| Frozen | embed_tokens, lm_head (tied), down_proj, all norms, base, SNAC, Vocos | Decision 2 / `model/freeze.py` |
| Trainable | **21.43 M params (0.649%)** — verified by the freeze verifier | — |
| Loss | per-frame-position weighted CE on audio + `<\|end_of_speech\|>`; prompt masked | Decision 4 + D14 |
| Position weights | `[1.0, 0.7, 0.4, 0.4, 0.7, 0.4, 0.4]` (c0=1.0, c1=0.7, c2=0.4) | Decision 4 (confirmed) |
| Optimizer | AdamW (HF default), lr 1e-4, warmup 20 | standard LoRA SFT |
| Batch | per-device 1 × grad-accum 4 (eff. 4) | fits memory; tune up on the server |
| Steps | 3000 (smoke: 100) | first real run; revisit from the loss/retention curves |
| Precision / ckpt | bf16, gradient checkpointing, `use_cache=False` | memory |
| Determinism | seed 3407; deterministic algos (`warn_only`) | `train/determinism.py` |

## Data path (per step)

loader (decode → 24 kHz resample → gates → speaker via D15) → **compile** (SNAC encode → 7-token
interleave, dedup) → collator (`[prompt \| audio \| <eos_speech>]`, prompt masked) → mixture
(56% SPRINGLab / 24% Rasa / 15% Hindi / 5% English per batch — Decision 3/7). Eval clips are
excluded by `clip_key` holdout. See docs/DATA.md.

## Monitoring + safety

- **Checkpoints** (`save_steps=500`): LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`),
  `config_snapshot.yaml`, RNG state, and the retention-eval JSON at that step.
- **Regression-stop** (`train/callbacks.py`, Decision 3): retention WER on retention_hindi +
  retention_english every 500 steps; if it regresses > 15% relative to the pre-train baseline, it
  logs a warning and drops a `regression_detected` marker — never silently continues. (The WER eval
  fn is injected from the Phase-6 harness; inert in the smoke.)

## Commands

```bash
python scripts/train.py --smoke     # 100 steps / 20 samples — proves the pipeline
python scripts/train.py             # full run (server; set full_gpu: true)
```

## What I'd change with more resources

Sweep LoRA rank and the MLP layer band; ablate the per-codebook weighting and the stop-token-in-loss
(D14) against the longform over-generation rate; raise effective batch size on the server; wire the
`indic_normalizer` Marathi number engine so digit-heavy STEM text is spoken, not spelled.
