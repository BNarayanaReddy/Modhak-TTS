# DECISIONS

The judgment log — the deliverable. Decisions 1–10 are the committed decisions (verbatim
from the project brief); D11+ are decisions taken during implementation, each with the
alternatives considered, why this one, and what I'd change with more resources. Runtime
conflicts found during Phase 0 are in docs/RECON.md §4–5.

---

## Decision 1: Fine-tuning objective.
The base model is closed-voice named-speaker TTS. Its official model card states: no voice
cloning, style is preview-quality, no sound-effect tokens, content-specific performance
with a documented weakness on "unusual words." Marathi is a first-class training language
with two native voices (Anagha, Chinmay); the other 42 voices speak Marathi cross-lingually.
I choose Option 1: domain adaptation on Marathi educational STEM content, using existing
voices. Rationale: it targets the model's documented weakness ("unusual words") in a domain
aligned with Bodhan's education mission; does not require touching speaker vocabulary or
embeddings; and eliminates entire classes of failure modes (voice-manifold corruption,
style inconsistency).
Rejected: Option 2 (voice-specific quality lift — less unique judgment).
Rejected: Option 3 (new voice — not documented as supported; no ground truth for embedding
init). Rejected: Option 4 (multi-speaker expansion — scope creep on Option 3).
Anticipated failure modes: (1) forgetting on non-Marathi and non-STEM Marathi content;
mitigated by original-domain replay and locked retention buckets. (2) mispronunciation of
English loanwords; mitigated by explicit code-switch handling in the data pipeline.
(3) prosodic overfitting to a narrow source; mitigated by data diversity across topics and
voices.

## Decision 2: LoRA target.
Target: q_proj, v_proj on ALL layers (rank 32, alpha 64, dropout 0.05).
        up_proj, gate_proj on layers 8..24 (rank 32, alpha 64).
Frozen: embed_tokens, lm_head, down_proj, all norm layers, SNAC codec, Vocos vocoder.
Rationale:
  - Q/V adaptation adjusts routing/attention over the mixed text+speaker+audio context.
    Cheap and defensible.
  - MLP up/gate on middle layers changes content-to-acoustic transformation without
    touching early layers (low-level acoustic features) or the final layers closest to the
    LM head (which would corrupt codebook 0 priors that carry intelligibility).
  - lm_head frozen: audio token priors are the most sensitive tensor in an AR codec-LM.
    Untouched.
  - embed_tokens frozen: no new symbols. No new voice tokens. No new style tokens. If frozen
    embed causes convergence issues, we log and consider unfreezing at very low LR, but
    default is frozen.
(Runtime note: the checkpoint has `tie_word_embeddings: true`, so lm_head and embed_tokens
are the same tensor — freezing the embedding freezes the head automatically; the freeze
verifier asserts on the shared tensor. See RECON §2.)

## Decision 3: Anti-forgetting.
Replay: 20% of every training batch is non-Marathi. Split: 15% Hindi (linguistically
closest, highest forgetting risk), 5% English (checks general Llama backbone health).
Source: held-out reference clips from snorbyte or another public Indic-TTS dataset;
documented in DATA.md. Regression stop: fast retention eval every 500 steps on Hindi +
English buckets. Roll back if retention WER regresses by more than 15% relative.

## Decision 4: Loss.
Cross-entropy on tokens strictly between <|start_of_speech|> and <|end_of_speech|>. Prompt
tokens are context, contribute zero loss. Optional per-codebook-position weighting: positions
{0, 7, 14, ...} (codebook 0 across frames) get weight 1.0; positions in codebook 1 get weight
0.7; positions in codebook 2 get weight 0.4. Configurable and ablatable. Document rationale
(codebook 0 carries perceptual weight disproportionate to bitrate).
(Refined by D14 on the stop token — see below.)

## Decision 5: Evaluation config.
Matches production: temperature 0.6, top_p 0.9, repetition_penalty 1.2, max_new_tokens 2520
(~30s). Vocos decoding for primary metrics. A/B stock SNAC decode on 10% of eval samples to
attribute failures between LM/vocoder.

## Decision 6: Style.
Scoped OUT. Empty style string during training. Do not condition on measured f0_std or
expressiveness scores from voices.md as a style prompt. Rationale: style is documented as
preview-quality; fighting it during fine-tune destabilizes the primary objective.

## Decision 7: Data mixture strategy under the closed-voice constraint.
The reasoning about Approach A vs. B vs. C, why the ideal (Anagha/Chinmay reading STEM
content) doesn't exist publicly, why the chosen hybrid (SPRINGLab as domain signal, Rasa as
voice anchor) is the defensible middle ground, and what tradeoff you're accepting (voice
drift toward SPRINGLab timbre in exchange for domain adaptation signal). Include the specific
anti-drift mechanism (Rasa Marathi replay at 30% of the Marathi batch, retention_rasa_marathi
eval bucket).

## Decision 8: Rejection of snorbyte pre-encoded data.
Note it was SNAC-encoded for a different upstream model, requires token remapping that the
community fine-tune repo documented, has documented gender imbalance in the Marathi split.
Chose to re-encode raw audio through the correct SNAC pipeline instead. This is a "no"
decision worth documenting because it demonstrates you looked at the obvious shortcut and
rejected it for a specific reason.

## Decision 9: What "STEM domain adaptation" actually means for this project.
The honest reframe: since STEM-specific Marathi audio in the target voices doesn't exist, the
training is really "Marathi quality lift" using SPRINGLab general Marathi as the signal, with
STEM appearing primarily in the evaluation target (adversarial bucket). Explicitly state
you're not overpromising a domain-specialist model.

## Decision 10: License stance across data sources.
Which sources you use, their licenses, whether the aggregate is redistributable, what
attribution is required. (Concrete license table lands in docs/DATA.md §1 once each source's
license is confirmed; the license gate rejects any clip whose license is unknown.)

---

## D11: SNAC token-space assertion (corrects GOAL Phase-0 step 4).
**What.** GOAL step-4 asserts `<|snac_k|> == base + k*4096` for k=0..6. Runtime (RECON
conflict A): the audio tokens are named PER-CODE — `<|snac_N|> == base + N` for N=0..28671 —
so that assertion is checking a naming scheme that does not exist. `tokens.assert_snac_contract`
instead verifies the real contract: base == `<|snac_0|>`, per-code contiguity across the
band, band top `<|snac_28671|>` == base+28671 == `<|speaker>`−1, and the 7 frame-position
bases `base + p*4096` all in-band.
**Alternatives.** Keep GOAL step-4 verbatim and let it fail (rejected — the check would
assert nothing true and block startup).
**Why.** The underlying contract the reference relies on (`bodhan_genai/tts/codec/snac.py`,
`ids_to_codes`) is arithmetic offsets from a single base; the corrected assertion guards
exactly that. Confirmed against the real vocab (test passes).
**More resources.** Add the full content-fingerprint from `token_contract.md` §6 and assert
it at train/eval startup, not just the layout.

## D12: Reference provenance.
**What.** `reference/inference.py`, `reference/token_contract.md`, `reference/voices.md` are
vendored verbatim from the `bodhan-ai/indic-speak` model repo and treated as authoritative,
do-not-edit. Our collator/generator match them.
**Alternatives.** Wait for a separately-provided file (none was); reimplement from the model
card (higher divergence risk).
**Why.** The published repo copy is the authoritative source; vendoring pins it so downstream
parity tests have a fixed oracle.
**More resources.** Track the upstream file's revision hash and alert on change.

## D13: Round-trip validation clip.
**What.** The Phase-0 SNAC↔Vocos round-trip used an English LibriSpeech clip
(`Narsil/asr_dummy/1.flac`), not Marathi.
**Alternatives.** Block Phase 0 on a Marathi clip.
**Why.** SNAC/Vocos is an acoustic codec — language-agnostic — so codec-wiring validation
(mel-corr 0.973) does not depend on language; authentic Marathi needed the `datasets` loader
(broken by a pyarrow incompatibility, RECON E2) or gated repos. A Marathi baseline was
separately captured at Phase-0 step 6 (Anagha/Chinmay generations).
**More resources.** Re-run the round-trip on a held-out Marathi clip once the compile step is
wired.

## D14: Stop-token in the training loss (refines Decision 4).
**What.** The collator includes `<|end_of_speech|>` in the loss (labeled, not −100),
alongside the audio tokens; the prompt (incl. `<|start_of_speech|>`) stays masked, and no
`<|end_of_ai|>` is appended (matches `reference/inference.py`, which stops at
`<|end_of_speech|>` — resolves RECON conflict D). Exposed as `include_stop_in_loss`
(default True).
**Alternatives.** (a) Decision 4 literal — audio tokens only, stop token masked; (b) full-
sequence loss (the `bodhan_genai` reference collator) — loss on the prompt too.
**Why.** Phase-0 showed the base model over-generates on Marathi STEM (8.5 s runaway, no
`<|end_of_speech|>`) while stopping cleanly on Hindi (1.0 s). If the stop token is masked, the
fine-tune never reinforces *when to stop*, so the exact failure we target goes unaddressed.
Keeping the prompt masked preserves Decision 4's core intent (don't re-learn text priors).
**More resources.** Ablate stop-in-loss vs. not; report over-generation / max_new_tokens-hit
rate on the longform bucket for each.

## D15: Speaker assignment — gender → target voice.
**What.** The source corpora carry no speaker names, only gender, so we map gender → the
language's library voice (configs/data.yaml `speaker_assignment`): SPRINGLab MR `0`→Anagha /
`1`→Chinmay; Rasa MR Female→Anagha / Male→Chinmay (the real studio anchor recordings); Hindi
`0`→Kavya / `1`→Amit; English→Amit (fixed; no gender column). Operationalizes Decision 7
(SPRINGLab general Marathi trained *as* the target voices, accepting drift; Rasa as the
genuine anchor).
**Alternatives.** (a) one target voice for all SPRINGLab (no gender split); (b) exclude
SPRINGLab from voiced training, Rasa-only (loses the domain signal); (c) a neutral/placeholder
speaker (violates the closed-voice constraint).
**Why + three tightenings.**
  1. **Gender-preserving.** The map never crosses gender — female sources → Anagha (female),
     male → Chinmay (male) — so any timbre drift stays *within* gender: a register/diction
     nudge, never a gender flip, and the source and target pitch statistics stay compatible.
     This bounds what "accepting drift" can cost perceptually.
  2. **Symmetric drift monitoring.** Drift is tracked on BOTH target voices, not just the
     female one: the eval computes a per-voice speaker-similarity trajectory against Rasa
     anchor clips for Anagha AND Chinmay, and both appear in the retention buckets — so
     asymmetric degradation (e.g. the male voice drifting more if its SPRINGLab share differs
     in quality) is caught, not averaged away.
  3. **SNR / bandwidth dead-man's-switch.** SPRINGLab is general-quality Marathi (variable
     SNR, possibly band-limited despite a 48 kHz container). The loader measures per-clip SNR
     and effective bandwidth; if a source's sampled clips fall below the floors
     (`gates.soft.snr_floor_db`, a wideband-fraction floor), the compile/train run trips a
     dead-man's-switch and halts rather than silently baking noise/band-limiting into
     Anagha/Chinmay. Concrete floors are set from the Phase-2 validation stats (below) before
     the mixture is locked.
**More resources.** Speaker verification/diarization to confirm SPRINGLab is single-speaker-
per-gender; per-utterance quality weighting rather than a hard switch.

## D16: marathi_code_switch eval bucket left empty.
**What.** The `marathi_code_switch` eval bucket is locked with 0 clips.
**Alternatives.** (a) Curate a synthetic Marathi+English probe set (synthesis-only, like
longform); (b) provide hand-written examples.
**Why.** The SPRINGLab and Rasa Marathi transcripts are pure Devanagari — no romanized
English loanwords — so there is no natural code-switch data to hold out. Rather than
fabricate a probe and risk over-claiming coverage, the bucket is left empty and the gap is
documented: the English-loanword mispronunciation failure mode named in Decision 1 is **not
measured** by the current eval set. This is the Decision-9 discipline — refusing to claim
more than the data supports.
**More resources.** Source a genuine Marathi code-switch corpus (spontaneous/conversational
speech carries romanized tech terms), or curate + record a small in-voice probe set.

## D17: Full-run training-data plan (shards, voice balance, mixture window).
**What.** The server full run (`scripts/train.py`, no `--smoke`) compiles a per-pool corpus and
draws it with the fixed per-batch composition (replay.py). Concretely:
- **Pools (unique clips, ≈ proportional to the mixture → ~4.8 epochs each over 3000 steps ×
  eff-batch 4 = 12 000 draws):** `springlab_marathi` 1400, `rasa_marathi` 600, `hindi_replay`
  375, `english_replay` 125. Counts live in `configs/train.yaml full.clips`; shard layout in
  `scripts/train.py FULL_PLAN`.
- **Voice balance.** The corpus shards are *gender-homogeneous* (verified: SPRINGLab MR shard 0 =
  gender0/Anagha, shard 3 = gender1/Chinmay; Rasa MR test0-3 Male/Chinmay, test4-6 Female/Anagha;
  SPRINGLab Hindi tr0-3 gender0/Kavya, tr4-9 gender1/Amit). So each pool explicitly lists the
  shards it needs: SPRINGLab MR uses shards 0+3; the **Rasa anchor is 50/50** across test2/3
  (Chinmay) + test4/5 (Anagha) so drift is monitored on both voices (D15 tightening #2) — *not*
  the Chinmay-only pool a naive test2-4 pick would have produced.
- **Hindi replay = Amit.** `retention_hindi` is voiced by Amit, so the replay that protects it uses
  Amit shards (tr4/5, gender1). Kavya-Hindi isn't measured, so it isn't replayed. English replay is
  Amit (fixed; no gender column).
- **Mixture window = 20 (`configs/train.yaml full.mixture_window`).** The per-batch composition is
  guaranteed over a 20-draw window, not the 4-example effective batch: at eff-batch 4, largest-
  remainder rounding gives {springlab 2, rasa 1, hindi 1, **english 0**} — English's 5% rounds away
  and it never trains. A 20-window seats all four {11,5,3,1} = 55/25/15/5 (exactly 20% non-Marathi
  replay, Decision 3), and English recurs at least every ~39 draws. The trainer's sequential sampler
  walks the order in eff-batch groups, so each window spans whole optimizer steps.
- **Disjoint from eval.** Rasa training shards (test2-5) don't overlap the eval shards (test0/1); the
  ~220 held-out eval clips are additionally excluded by key at compile time (`reject_eval_bucket_*`).
- **GPU SNAC compile.** On `full_gpu`, SNAC loads on CUDA and encodes on the A100 (decode/resample
  stay on CPU) — the smoke path ran SNAC on CPU.

**Alternatives.** Precompile the corpus to an on-disk cache (faster re-runs; deferred — a first run
compiles in-process). Rasa *train* shards (57, far larger) instead of *test* (chose test: same two
studio speakers, small, adjacent to the eval distribution, disjoint by shard). Larger pools / more
steps (3000 steps on ~2500 clips ≈ 5 epochs is a defensible LoRA domain-adaptation budget; 500-step
checkpoints let eval pick an earlier one if it plateaus).
**Why.** Correctness of the *training signal* over raw scale: the anchor must be voice-balanced or the
symmetric-drift guarantee is a fiction; replay must match the capability we measure or it protects the
wrong thing. Scale sits in config so it's a one-line change, not a code edit.

## CONFLICT (eval, flag before Phase 6): locked `retention_rasa` bucket is Chinmay-only.
**Found at runtime.** D15 tightening #2 states Anagha's drift is tracked "against Rasa anchor clips
for Anagha AND Chinmay." But the locked eval buckets drew Rasa only from test0/test1, which are
**both Male/Chinmay** — so the eval set contains **no Anagha Rasa reference clips**. Anagha's anchor
retention therefore cannot be measured against the *true* Rasa anchor; only Chinmay can. (Anagha drift
is still observable via the SPRINGLab `general_unseen` references, a weaker anchor.)
**Does not affect training** — the training anchor above IS voice-balanced. This is an eval-coverage
gap in the *locked* artifact. Not resolved silently: buckets are locked, so fixing means a re-lock
(add an Anagha Rasa reference sub-bucket from test4/5) — a call for before the eval run, reported to
the user, not made unilaterally.

## D17 — run outcome + loss-display note (server full run, 2026-09-24).
**Outcome.** Completed on the A100-40GB: 3000 steps, 12 000 mixture draws over 2500 unique clips
(springlab 1400 / rasa 600 / hindi 375 / english 125), 21 430 272 trainable params, ~103 min.
Adapter at `artifacts/checkpoints/main_run/` (adapter_model.safetensors + adapter_config.json +
config_snapshot.yaml + rng + checkpoint-3000). Per-token weighted-CE fell **4.25 → 3.15** (monotonic;
most of the drop by step 500, then gradual — a healthy LoRA domain-adaptation curve, below the smoke's
3.65). Grad-norm stayed ~5–6, no divergence.
**Loss-display note (readability, not a bug).** `weighted_audio_ce` returns a weighted MEAN
(`(ce·w).sum()/w.sum()` ≈ per-token CE ≈ 3–4). But HF Trainer logs a *custom* `compute_loss`
accumulated over the grad-accum window, so the *logged* number is ≈ grad_accum× the mean: the smoke
(grad_accum=1) showed ~4.4→3.65, this run (grad_accum=4) shows ~17→12.6 for the *same* underlying
loss. Optimization is unaffected — gradients are averaged correctly (smooth descent + stable grad-norm
confirm), only the printed scalar is scaled. `training_log.txt` now records `final_loss_per_token`
(logged ÷ grad_accum) as the honest, cross-run-comparable figure. A cleaner fix (thread
`num_items_in_batch` through so the Trainer's own display is exact) is a follow-up, not worth
re-running a good adapter for.

## D17 CONFLICT — RESOLVED (re-lock, 2026-09-24).
The Chinmay-only `retention_rasa` gap is fixed by `scripts/relock_add_anagha_anchor.py`: +30 general
Anagha clips from Rasa **test6** (a Female shard the training run never used → zero leakage). The bucket
is now 30 Chinmay + 30 Anagha (both with reference audio); only `retention_rasa_marathi.json`,
`held_out_clip_keys.json` (220→250) and `_summary.json` changed — the other 7 locked buckets are
byte-identical. The re-lock is an additive, idempotent amendment layered on the original build (kept
canonical), the right way to correct a locked artifact. Symmetric anchor-drift monitoring (D15 #2) is
now backed by real per-voice Rasa references.

## D18: Phase 6 eval wiring (what the running model actually needed).
Bringing the eval up on the server surfaced several gaps between the scaffolded code and the real
backends — fixed against the running models, not guessed:
- **ASR interface was wrong.** `asr.py` assumed `AutoModel`/`AutoProcessor` on a numpy array. The real
  `indic-transcribe-core` (FastConformer/canary-1b-v2, gated) ships its own code: a callable
  `IndicTranscribe.from_pretrained(<local dir>)` that transcribes a FILE PATH (its SentencePiece
  tokenizer is path-based). Rewrote the wrapper to snapshot the repo (skipping the 2.5 GB `nemo/`
  checkpoint the HF path doesn't use), put the dir on `sys.path`, and stage each waveform to a temp
  wav. Verified: near-perfect Marathi transcription under transformers 5.14.1.
- **Per-item ASR language (caught by the smoke).** The ASR language was fixed to `mr` for every bucket,
  so `retention_english`/`retention_hindi` were transcribed by a Marathi ASR — English WER came back at
  1.54 (nonsense). Each eval item carries its `language_id`; the panel now transcribes every clip in its
  own language (hi/en/mr). Without this the retention metric — the entire point of those buckets — is
  invalid.
- **Reference audio didn't exist.** `run_panel` needs a `reference_audio_fn` for speaker similarity but
  none was implemented or passed, so speaker sim silently never ran. Added `eval/reference.py`
  (`ReferenceAudioProvider`): resolves each held-out clip back to its real 24 kHz waveform, streaming
  each shard once (keyed by the manifest's `clip_key`, early-stopping when found). Wired into `eval.py`.
- **Per-voice speaker similarity (from the D17 re-lock).** The panel averaged similarity over a whole
  bucket; now it also reports `speaker_similarity_by_voice` so Anagha and Chinmay are tracked separately
  (D15 #2). Embedder discriminates them cleanly on real refs: ceiling ~0.53–0.64 vs floor ~0.21.
- **Server deps (via uv, torch untouched):** torchaudio, sentencepiece (ASR); speechbrain, hyperpyyaml,
  ruamel.yaml==0.18.6 (embedder — 0.19 breaks hyperpyyaml's Loader). Added an eval `--limit` flag for
  the smoke pass that caught the language bug.
**Why.** These are the reference-parity discipline applied to eval: the eval is only a deliverable if its
metrics are valid, so the backends were verified against the real models before trusting any number.
