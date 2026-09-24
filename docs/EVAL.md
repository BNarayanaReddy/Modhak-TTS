# EVAL — metric vector, model choices, protocol

Even though the evaluator isn't scoring metrics, the harness is a first-class deliverable —
it's the strongest signal of judgment. Run it on any checkpoint: `python scripts/eval.py
--label baseline` (base) and `--label finetuned --adapter <dir>`. Config: `configs/eval.yaml`.

## Metric vector (per locked bucket)

| Bucket | Primary metric | Also |
|---|---|---|
| marathi_adversarial | WER (numbers/proper-nouns/STEM — the weak surface) | ins/del/sub, max-hit rate |
| marathi_general_unseen_domain | WER | ins/del/sub |
| marathi_stem_seen_voice | WER + **speaker similarity** (vs ceiling/floor) | prosody |
| retention_rasa_marathi | speaker similarity (anchor retention) + WER | prosody |
| retention_hindi (Amit) | **WER** — the single most important post-FT metric | ins/del/sub |
| retention_english | WER (backbone health) | ins/del/sub |
| marathi_longform | n-gram repetition (3/5-gram), max-hit rate, similarity trajectory | drift |
| marathi_code_switch | — (empty; D16) | — |

## Per-metric rationale

- **WER, with insertions/deletions/substitutions split.** Substitutions ≈ mispronunciation
  (the "unusual words" weakness); insertions/deletions ≈ runaway/truncation. Splitting them tells
  you *which* failure moved. Raw hypotheses are persisted to `eval_outputs/` for audit.
- **Retention WER (Hindi/English)** is the guardrail: domain adaptation must not degrade the
  other languages (Decision 3). The training regression-stop watches exactly this.
- **Speaker similarity, calibrated** with a **ceiling** (two genuine same-voice clips) and a
  **floor** (Anagha vs Chinmay). A bare cosine is uninterpretable; the anchors turn it into
  "voice preserved" vs "drifting toward a different voice" — the D15 drift, tracked on **both**
  target voices.
- **Longform repetition + max-hit rate** measure the exact base-model failure Phase 0 surfaced
  (8.5 s Marathi runaway) — the thing D14's stop-token-in-loss is meant to fix.
- **Prosody** (F0 mean/std, energy std, pause ratio) catches monotone/over-expressive drift that
  WER misses.

## Model choices

- **Marathi ASR: `bodhan-ai/indic-transcribe-core`** (confirmed) — Bodhan's own ASR, purpose-built
  for these languages; using the same ecosystem's ASR to score its TTS is the honest, aligned
  choice. Gated (accept terms), custom "indic-canary" arch, `trust_remote_code`.
- **Speaker embedder: `speechbrain/spkrec-ecapa-voxceleb`** — standard ECAPA-TDNN; only relative
  (calibrated) scores are reported, so the absolute embedder identity matters less than the anchors.
- **Vocoder: Vocos (primary)** per the reference; **stock SNAC decode A/B** on 10% of samples
  (`eval/stock_vs_vocos.py`) attributes a regression to the LM vs the vocoder (Decision 5).

## Statistical protocol

- **95% bootstrap CIs over utterances** (`eval/metrics.bootstrap_ci`, 1000 resamples) on every WER
  bucket — small locked buckets (30–40 clips) need CIs, not point estimates, to avoid over-reading
  noise. Per-utterance WER is the resampling unit.
- Generation matches production (Decision 5): temp 0.6, top_p 0.9, **rep-pen 1.2**, max_new 2520.
  (Reference *parity* uses rep-pen 1.0 — the reference script applies none; see `inference/generate.py`.)
- Report: `eval/report.py` → theme-aware HTML with per-bucket tables + CIs + the raw JSON.

## What I'd change with more resources

MOS / CMOS listening tests (WER is a proxy for intelligibility, not naturalness); a larger,
professionally-transcribed Marathi STEM eval set; CI on the speaker-similarity and prosody deltas,
not just WER.
