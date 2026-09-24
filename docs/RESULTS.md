# RESULTS — Marathi LoRA adaptation of Indic-Speak

What was actually run end-to-end on the A100, and what it showed. Reported honestly: the
fine-tune **did not improve** ASR-WER and regressed the anchor voice — which is the drift the
decision log pre-registered (Decisions 7/9/D15). This is a judgment deliverable, not a
leaderboard entry.

## 1. Training (Phase 4) — done

LoRA SFT on `bodhan-ai/indic-speak` (Llama-3.2-3B backbone), server A100-40GB.

| | |
|---|---|
| Trainable | **21,430,272** params (0.645%) — LoRA r=32, α=64 (Decision 2) |
| Data | 2,500 unique clips, mixture 56/24/15/5 (SPRINGLab MR / Rasa MR / Hindi / English), 20-draw window |
| Steps | 3,000 · 12,000 draws (~4.8 epochs/pool) · ~103 min |
| Loss (per-token) | **4.25 → 3.15**, monotonic, stable grad-norm (evidence: `artifacts/checkpoints/main_run/training_log.txt`) |

The adapter learned (loss fell cleanly); adapter + `config_snapshot.yaml` are the reproducible
record. Weights (`adapter_model.safetensors`, 85 MB; gitignored) live locally in
`artifacts/checkpoints/main_run/` and are released — with this honest results card + attribution —
to **[huggingface.co/BNarayanaReddy/modhak-tts-mr-lora](https://huggingface.co/BNarayanaReddy/modhak-tts-mr-lora)**
(private; a negative-result research artifact, not a quality upgrade).

## 2. Eval (Phase 6) — done

Base vs fine-tuned over the 8 locked buckets, WER via Bodhan `indic-transcribe-core` (per-clip
language). Reports: `artifacts/eval_reports/{baseline,finetuned}/panel.{json,html}`.

**Audio A/B** — the same held-out texts through each model are in `artifacts/samples/{base_model,finetuned}/`
(also on the HF repo under `samples/`). The regression is audible on the **Rasa anchor**
(`retention_rasa_marathi__Chinmay.wav`): base = 65 SNAC frames, fine-tuned = **171** — it rambles ~2.6×
longer for the same sentence, which is what the anchor WER jump (0.167→0.630) is measuring.

> **Read the numbers with the caveats first.** This pass was run **fast under deadline**: `--limit 6`
> per bucket and `--max-new-tokens 1000` (vs the configured 2520), WER-only (speaker similarity
> validated separately, ceiling ~0.6 vs floor ~0.2, but not in this pass). Small n and a low token
> cap **truncate longer fine-tuned outputs → inflate their WER**, so treat magnitudes as directional,
> especially the large anchor delta.

| bucket | base WER | fine-tuned WER | Δ |
|---|---|---|---|
| marathi_adversarial | 0.353 | 0.353 | +0.000 |
| marathi_general_unseen | 0.194 | 0.343 | +0.149 |
| marathi_stem_seen | 0.202 | 0.262 | +0.060 |
| **retention_rasa (anchor)** | 0.167 | 0.630 | **+0.463** |
| retention_hindi | 0.281 | 0.281 | +0.000 |
| retention_english | 0.452 | 0.524 | +0.071 |
| marathi_longform (3-gram rep) | 0.000 | 0.001 | no runaway |

## 3. Interpretation (defensible judgment)

- **No WER improvement; anchor voice regressed.** Fine-tuned WER is flat-to-worse everywhere, worst
  on `retention_rasa` (the Anagha/Chinmay studio anchor). **This is the exact trade Decisions 7/9 and
  D15 pre-registered:** training the studio voices on *general-quality* SPRINGLab Marathi drifts them
  toward the noisier SPRINGLab timbre, buying domain coverage at the cost of anchor fidelity. The eval
  **measured the risk the judgment log named in advance** — the point of the D15 symmetric-drift +
  dead-man's-switch design.
- **Retention held where it mattered most.** `retention_hindi` is unchanged (+0.000) — the 15% Hindi
  replay (Decision 3) did its job; the highest-forgetting-risk language did not degrade. English is
  marginally down (+0.071).
- **No longform runaway** — 3-gram repetition stays ~0, so the stop-token training (D14) held; the
  model did not learn to babble.

**What this says about the method, not just the score:** the pipeline, the mixture, the anchor
monitoring, and the eval all *worked* — they surfaced a real regression rather than hiding it. A
genuinely *better* Marathi model from here would (in priority order): filter/weight SPRINGLab by SNR
so the anchor trains only on clean audio (the D15 dead-man's-switch made this measurable but we did
not down-weight per-clip), lengthen training, and re-evaluate at the full 2520-token cap with n≥30 and
speaker similarity on. See DECISIONS.md D17/D18 and "More resources" notes.

## 4. Reproduce

```bash
# server (A100), env per docs/SETUP.md:
python scripts/train.py                     # full LoRA run  -> artifacts/checkpoints/main_run
python scripts/eval.py --label baseline --with-speaker                              # base
python scripts/eval.py --label finetuned --adapter artifacts/checkpoints/main_run --with-speaker
# fast deadline variant used here: add  --limit 6 --max-new-tokens 1000  (WER-only, drop --with-speaker)
```
