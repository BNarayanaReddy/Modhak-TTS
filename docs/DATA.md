# DATA — pipeline reference

Operational reference (facts + configs). Reasoning lives in DECISIONS.md; this doc points
to it. Config of record: `configs/data.yaml`. Confirmed 2026-09-24.

## 1. Sources

| Source (HF) | License | Role | Lang | Speakers (via D15) | Notes |
|---|---|---|---|---|---|
| SPRINGLab/IndicTTS_Marathi | CC-BY-4.0¹ | primary training (domain signal) | mr | Anagha (g0/F), Chinmay (g1/M) | 4 shards, ~10.9k rows, 48 kHz; gender-sharded |
| ai4bharat/Rasa → `Marathi/` | CC-BY-4.0¹ | voice anchor + eval | mr | Anagha (F), Chinmay (M) | gated; has `style` + `duration`; 48 kHz |
| SPRINGLab/IndicTTS-Hindi | CC-BY-4.0¹ | replay + retention | hi | Kavya (g0/F), Amit (g1/M) | 48 kHz |
| SPRINGLab/IndicTTS-English | CC-BY-4.0¹ | replay + retention | en | Amit (fixed) | no gender column; 48 kHz |
| ~~IndicVoices-R Marathi~~ | — | — | — | — | **dropped** — no such public dataset (RECON conflict; Decision 9) |

¹ License to be confirmed against each repo's card before redistribution (Decision 10). The
loader's license gate rejects any clip whose license is unknown. Rationale for the source
choice: Decisions 7, 8, 9. All are 48 kHz → resampled to 24 kHz (SNAC) by the loader.

## 2. Schema (per clip — `src/indic_speak_ft/data/schema.py`)

| Field | Type | Source |
|---|---|---|
| provenance.source_id / license | str / str | provenance |
| audio.sample_rate / duration_s / snr_estimate / clipping_ratio | int / float / float? / float | measured |
| transcript.text_raw / text_normalized / normalizer_version | str / str / str | provenance / measured |
| text.language_id (mr/hi/en) / script (Deva/Latn) | enum | labeled / measured |
| text.num_digits / num_english_words / code_switch_spans / domain (stem/general) | int / int / list / enum | measured |
| speaker.name (∈ 44-voice library) / gender (F/M) | str / enum | labeled (D15) |
| style.mean_f0 / f0_std | float? / float? | measured — METADATA ONLY (Decision 6) |
| dedup.audio_hash / transcript_shingle_hash | str / str | measured |

Eval-holdout uses a separate cheap `clip_key = sha16(source::audio.path)` (loader.eval_clip_key).

## 3. Gates (`configs/data.yaml` → `gates`)

**HARD (drop):** license unknown · duration ∉ [0.5, 20.0] s · empty text_normalized ·
speaker ∉ 44-voice library · audio in a locked eval bucket (`clip_key` holdout).
**SOFT (flag, keep):** SNR < 15 dB · clipping_ratio > 0.01 · bandwidth < 6 kHz (narrowband).

**Dead-man's-switch (D15):** pre-flight per-source sample (64 clips) — halt if median SNR
< 20 dB **or** wideband fraction < 0.90. Validation: Anagha ~87 dB / Chinmay ~93 dB SNR,
0% narrowband → both pass. Thresholds justified in DECISIONS.md D15.

## 4. Mixture (per batch — `configs/data.yaml` → `mixture`)

| Source | Fraction | Derivation |
|---|---:|---|
| SPRINGLab MR | 0.56 | 0.80 Marathi × 0.70 (Decision 7) |
| Rasa MR (anchor) | 0.24 | 0.80 Marathi × 0.30 Rasa (Decision 7) |
| Hindi replay | 0.15 | Decision 3 |
| English replay | 0.05 | Decision 3 |

Per-batch composition is EXACT (largest-remainder; `mixture.batch_composition`), so replay
never rounds to zero — Decision 3 is a per-batch guarantee.

## 5. Eval buckets (locked — `configs/eval_manifests/*.json`)

8 disjoint buckets; every ref clip held out from training via `clip_key`
(`held_out_clip_keys.json`). Counts filled by `scripts/build_eval_buckets.py`.

| Bucket | Tests | Source → Voice | Ref audio |
|---|---|---|---|
| marathi_adversarial | numbers/dates/proper-nouns/STEM (weak surface) | Rasa NAMES/PROPER-NOUN + digits | yes |
| marathi_code_switch | English loanwords / code-switch | MR w/ Latin words | yes |
| marathi_stem_seen_voice | Anagha/Chinmay on STEM-adjacent text | Rasa WIKI/INDIC | yes |
| marathi_general_unseen_domain | general MR quality | SPRINGLab MR | yes |
| retention_rasa_marathi | anchor voice retention | Rasa MR general | yes |
| retention_hindi | Hindi forgetting (closest) | Hindi → **Amit** | yes |
| retention_english | backbone health | English → Amit | yes |
| marathi_longform | 30–60 s drift/repetition | synth from held-out MR text | no (synthesis-only) |

**`marathi_code_switch` is empty (count 0):** the SPRINGLab/Rasa Marathi transcripts are pure
Devanagari — no romanized English loanwords — so there is no natural code-switch data to hold
out. Left empty by decision (D16); the loanword-mispronunciation failure mode named in
Decision 1 is therefore **not measured** by this eval set. Locked bucket counts (seed 1234):
adversarial 40 · stem_seen_voice 40 · general_unseen_domain 40 · rasa_retention 30 ·
retention_hindi 40 · retention_english 30 · longform 12 · code_switch 0 · 220 clips held out.

## 6. Text normalization (`src/indic_speak_ft/data/text_norm.py`, `mr-norm-v1`)

NFC · preserve punctuation incl. danda ।/॥ · **no lowercasing** · Latin runs kept verbatim
and marked as `code_switch_spans` (offsets into text_normalized) · digits counted (ASCII +
Devanagari ०–९) · number→words expansion pluggable (TODO logged until the Bodhan
`indic_normalizer` Marathi engine is wired) · `text_raw` never mutated.

## 7. Reproduction

```bash
# validate the loader end-to-end + SPRINGLab per-speaker quality (streamed, no full download)
python scripts/validate_loader.py
# (re)build + lock the eval-bucket manifests (light columns only)
python scripts/build_eval_buckets.py
```

Full corpora (6–78 GB each) are not downloaded locally; the compile/train step runs on the
server (docs/SETUP.md). The manifests and held-out keys are the committed, reproducible lock.
