# OVERVIEW — how the pipeline is built, phase by phase

The build follows the GOAL phase order (0–7); each phase produced a testable slice of
`src/indic_speak_ft/` plus a doc. This is the "how it's structured" reference — start from the
[README](../README.md) for navigation, [RESULTS.md](RESULTS.md) for the outcome, and
[DECISIONS.md](../DECISIONS.md) for the judgment calls (D-numbers referenced below).

## Phase 0 — recon
Token contract resolved live against the real tokenizer (`configs/tokens_resolved.yaml`);
**SNAC↔Vocos round-trip** verified (mel-corr 0.973); reference baseline + determinism reproduced.
Runtime conflicts vs the docs recorded in [RECON.md](RECON.md).

## Phase 1 — token contract
`tokens.py` — frame math + the 7-token SNAC interleave, byte-for-byte inverse of
`reference.ids_to_codes` (guarded by tests).

## Phase 2 — data
`data/` — pydantic schema, quality gates, the D15 speaker map, Marathi text normalization, the
mixture sampler, and the collator. **8 eval buckets locked** in `configs/eval_manifests/` with the
held-out clip keys. Loader validated on the real corpora; SPRINGLab per-speaker quality checked.
See [DATA.md](DATA.md).

## Phase 3 — model / LoRA / freeze
`model/` — LoRA target selection (Decision 2), the freeze policy, and a verifier that asserts the
trainable set: **21,430,272 params (0.645%)**; base, tied embeddings/lm_head, SNAC, Vocos all frozen.

## Phase 4 — training
`train/` + `scripts/train.py` — SNAC compile, per-frame-position weighted loss (D4 + D14), replay
ordering, determinism, and the `WeightedLoRATrainer`. Runs on the server (the 11 GB laptop OOMs under
bf16 + offload). Recipe: [TRAIN.md](TRAIN.md). The full run's data plan + mixture window are D17.

## Phase 5 — inference
`inference/` + `scripts/sample.py` — `generate.py` (reference generation semantics + adapter
attach/merge) and `decode.py` (SNAC + Vocos → wav). Reference parity tested (de-interleave + prompt
byte-for-byte).

## Phase 6 — eval
`eval/` + `scripts/eval.py` — per-language WER (Bodhan `indic-transcribe-core`), per-voice speaker
similarity (ECAPA + real reference audio), prosody, retention, longform repetition, stock-vs-Vocos
A/B, panel → HTML report. The real-model wiring corrections found on the first server run are D18.
See [EVAL.md](EVAL.md) and the outcome in [RESULTS.md](RESULTS.md).

## Module map

| path | what it holds |
|---|---|
| `src/indic_speak_ft/tokens.py` | token contract + SNAC frame interleave |
| `src/indic_speak_ft/voices.py` | the 44-voice roster |
| `src/indic_speak_ft/data/` | schema, gates, speaker map, text-norm, mixture, collator, compile, dataset |
| `src/indic_speak_ft/model/` | LoRA config, freeze policy + verifier, base/SNAC/Vocos loading |
| `src/indic_speak_ft/train/` | weighted loss, replay, callbacks, determinism, trainer |
| `src/indic_speak_ft/inference/` | generate, decode |
| `src/indic_speak_ft/eval/` | metrics, asr, speaker, reference, prosody, retention, longform, panel, report |
| `scripts/` | recon_parity, validate_loader, build_eval_buckets, train, sample, eval (+ server runners) |
| `configs/` | base/data/lora/train/eval yaml + `eval_manifests/` (locked buckets) |
| `reference/` | authoritative, do-not-edit: inference.py, token_contract.md, voices.md |
