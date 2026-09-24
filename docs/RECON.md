# RECON — Phase 0 findings

Recon date: 2026-09-24. Target model: **bodhan-ai/indic-speak** (public on HF Hub).
Everything below is measured at runtime, not transcribed. Runtime code re-resolves
all IDs from the live tokenizer; this doc + `configs/tokens_resolved.yaml` are the
recorded snapshot.

## 0. Status summary

| Phase-0 step | Result |
|---|---|
| 1. Scaffold repo, pin deps | ✅ done (this repo; `pyproject.toml`) |
| 2. Load tokenizer | ✅ (via raw `tokenizer.json`; see env note E1) |
| 3. Resolve every dependent token ID | ✅ all resolve; map in `configs/tokens_resolved.yaml` |
| 4. Assert SNAC token space | ⚠️ **conflict A** — GOAL.md assertion is wrong as written; real contract holds |
| 5. SNAC↔Vocos round-trip | ✅ **PASS** (mel-corr 0.973, time-aligned) |
| 6. Reference baseline + determinism | ✅ **PASS** — LM loads, deterministic, documented example stops cleanly |

**Phase 0 is complete.** The round-trip (the check GOAL.md flags as load-bearing for
everything downstream) and the reference baseline both pass; the token contract is
resolved; the repo is scaffolded.

## 1. Environment (this machine)

| | |
|---|---|
| Host | narayana-LOQ, Ubuntu 22.04, Linux 6.8, x86_64 |
| Python | 3.10.12 |
| GPU | **NVIDIA RTX 3050 6 GB Laptop GPU**, driver present |
| torch | **2.13.0+cu130**, `cuda.is_available()==True`, 1 device |
| RAM | 11 GiB total (~2.8 GiB free at recon; heavy swap use) |
| Disk `/` | 147 G, **4.1 G free (98%)** — HF cache already 9.9 G |
| Offload `/media/narayana/Windows-SSD` | 298 G, **6.7 G free (98%)**, NTFS/fuseblk |

**Compute/storage plan (per your direction):**
- venv + code: `/home/narayana/ai/bodhan/Modhak-TTS`
- model downloads / offload: `/media/narayana/Windows-SSD/ai-models/bodhan_offload`
- **inference**: this laptop (6 GB GPU). 3.3B bf16 (~6.6 GB) > 6 GB VRAM, so LM
  inference here needs bf16 + CPU offload via `accelerate` (faithful math), **not**
  4-bit (would break parity). SNAC + Vocos fit easily.
- **training / fine-tuning**: on your server (not this machine).

Codec assets already staged on the offload drive:
`models/snac_24khz/` (77 MB), `models/indic-speak/vocos/best.pt` (456 MB).

### Environment issues found (need handling before their phases)
- **E1 — transformers version.** The model + tokenizer were saved with
  **transformers 5.14.1** (`tokenizer_class: TokenizersBackend`, `generation_config`
  `transformers_version: 5.14.1`). Installed system-wide is **4.46.3**, which cannot
  load the tokenizer via `AutoTokenizer` (`TokenizersBackend` unknown) nor accept the
  reference's `from_pretrained(..., dtype=...)` kwarg. Recon worked around it by
  reading `tokenizer.json` directly with the `tokenizers` library. **Fix:** a project
  venv pinned to `transformers==5.14.1` (isolated from the system 4.46.3). Required
  for steps 6, and all model loading. `transformers==5.14.1` is on PyPI.
- **E2 — datasets/pyarrow.** Installed `datasets==2.14.4` is incompatible with the
  installed `pyarrow` (`module 'pyarrow' has no attribute 'PyExtensionType'`), so
  `load_dataset(...)` fails. This blocks HF dataset loading (Phase 2). **Fix:** pin a
  newer `datasets` (>=3.x) in the venv. Not needed for Phase 0.
- **E3 — librosa/joblib.** `joblib` was missing (broke `librosa`); installed. Noting
  so the venv pins it explicitly.

## 2. Token contract (verified)

Full map in `configs/tokens_resolved.yaml`. All structural/control/conditioning
tokens resolve and match the model's own `reference/token_contract.md` exactly:

- Structural: begin_of_text=128000, eot_id=128009, start/end_of_human=128259/128260,
  start/end_of_ai=128261/128262, start/end_of_speech=128257/128258, pad=128263,
  eos(end_of_text)=128001.
- Conditioning: speaker 156938/156939, style 156940/156941, env 156942/156943.
- SNAC base `<|snac_0|>` = 128266; band 128266..156937 (28,672 = 7×4096).
- `tie_word_embeddings: true` → **lm_head shares storage with embed_tokens**. Decision 2
  freezes both; consistent (freezing the embedding freezes the head). The freeze
  verifier (Phase 3) should assert on the shared tensor, not two independent ones.

**Prompt template validated end-to-end.** Ran the *actual* reference
`build_prompt(speaker="Anagha", style="")` and got, in order: `<|start_of_human|>`
→ `<|begin_of_text|>` → `<|speaker>` → **"Anagha" as raw BPE text** (`An`/`ag`/`ha`,
ids 2127/351/4317) → `<speaker|>` → `\n` (198) → Marathi text BPE → `<|eot_id|>` →
`<|end_of_human|>` → `<|start_of_ai|>` → `<|start_of_speech|>`. Empty style emits
nothing. Speaker names are raw text, not dedicated tokens — confirmed.

## 3. Round-trip check (step 5) — PASS

Path: real speech → SNAC.encode → 7-token interleave (`bodhan_genai` `encode_audio`,
offset `base + position*4096`) → reference `ids_to_codes` → `quantizer.from_codes`
→ z_q [1,768,L] → **Vocos** (primary) and **SNAC.decoder** (stock A/B).

Clip: `Narsil/asr_dummy/1.flac`, 10.44 s, resampled to 24 kHz.
**Note:** this is an English LibriSpeech clip, used because the SNAC↔Vocos path is
language-agnostic (an acoustic codec) and authentic Marathi clips need `datasets`
(blocked, E2) or are gated. A Marathi clip is substituted once the Phase-2 dataset is
chosen. This does not weaken the codec-wiring validation.

Results:
- Token-contract assertions **all pass**: `len % 7 == 0`, all ids in `[base, base+28672)`,
  raw codes in `[0,4096)`, every frame-position-0 token in the codebook-0 band.
- Shapes correct: c0:c1:c2 = 107:214:428 (1:2:4), z_q [1,768,428].
- Encode rate ≈ 72 tok/s (vs the "82 tok/s" rule of thumb; lower because dedup drops
  consecutive-duplicate frames — 16 of 123 frames here).
- **Fidelity (dedup off, time-aligned): mel-corr 0.973**, exact duration match; Vocos
  vs stock-SNAC agree at 0.977. (A naive frame-aligned comparison *with* dedup shows a
  misleading 0.49 purely from the frame shift — documented so nobody re-derives it.)

Audio evidence: `artifacts/recon/{00_original_24k,01_roundtrip_vocos,02_roundtrip_stock_snac,03_nodedup_vocos}.wav`.

## 4. Conflicts (docs vs runtime) — reported, not silently resolved

**A. GOAL.md Phase-0 step-4 assertion is wrong as written.** It asserts
`<|snac_k|> == base + k*4096` for k=0..6. Runtime: audio tokens are named **per-code**,
`<|snac_N|> == base + N` (N=0..28671), so `<|snac_1|>` = 128267, not 132362. The
*underlying* contract GOAL.md/reference actually rely on is intact: single base token
`<|snac_0|>`, contiguous 7×4096 band, arithmetic per-frame-position offsets. Confirmed
against `bodhan_genai/tts/codec/snac.py` (offset = `base + position_index*4096`).
→ **Resolution (you approved using my judgment):** replace step-4 with the real
contract check — assert base==128266; band contiguous (`<|snac_N|>`==base+N sampled
across the band); top `<|snac_28671|>`==156937==`<|speaker>`−1; the 7 frame-position
bases `base+p*4096` all in-band. Implemented in Phase 1 `tokens.py` / `test_tokens.py`.

**B. Token-name typo in GOAL.md step-3 list.** It lists `<|/style>` (resolves to None).
The real close tag is `<style|>` = 156941 (already correct in GOAL.md's template block).
Low-stakes (style is scoped out, Decision 6); `tokens.py` resolves the real names.

**C. transformers ≥5 required.** See E1.

**D. `<|end_of_ai|>` (Phase-2 collator flag).** `token_contract.md` §5 and the Bodhan
training collator close the model turn with `<|end_of_ai|>` after `<|end_of_speech|>`,
but GOAL.md's template and the reference `build_prompt` stop earlier. The collator
must decide explicitly (see Phase-2 flags below).

**E. Round-trip clip language** — English substitute, see §3. Marathi pending Phase-2
dataset choice.

## 5. Flags for later phases (from the authoritative Bodhan source)

Pulled `Bodhan-AI/bodhan_genai` (the public stack GOAL.md points to) and read the
real `codec/snac.py`, `templates/chat.py`, `inference/prompts.py`. Two design
divergences to resolve deliberately — **not** silently:

- **F1 (Phase 4 — loss).** Bodhan's training collator uses **full-sequence loss**
  (`labels = ids[:]`, loss on prompt+audio). **Decision 4** specifies **audio-only
  masked loss** (labels −100 on the prompt). Decision 4 is a deliberate, defensible
  deviation for domain adaptation (don't re-learn text priors), but it *is* a
  divergence from the reference recipe. Flagged for your confirmation at Phase 4.
- **F2 (Phase 2 — prompt/target boundary).** Two conventions exist:
  reference `inference.py` puts `<|start_of_speech|>` at the END of the prompt (model
  generates audio directly); Bodhan `prompts.py` ends the prompt at `<|start_of_ai|>`
  and the model emits `<|start_of_speech|>` itself, with `<|end_of_ai|>` terminating
  the target. GOAL.md designates `inference.py` authoritative, so the collator will
  match the inference.py boundary; the `<|end_of_ai|>` question (D) is decided with it.
- **F3 (data).** SNAC encode removes consecutive-duplicate frames (same c0). Training
  targets and any duration bookkeeping must account for this (token count ≠ ∝ duration).
- **F4 (assets available).** `bodhan_genai` has ready `codec/snac.py` (encode/decode
  we already reused for the round-trip) and `templates/chat.py`; `indic_normalizer`
  has a Marathi (`mar/`) number engine + LaTeX/chemistry handling useful for STEM text
  norm (Phase 2). We can install/vendor these rather than reinventing.

## 6. Reference baseline + determinism (step 6) — PASS

Ran `scripts/recon_parity.py` (reproducible): loads the 6.6 GB LM as `LlamaForCausalLM`
on the installed **transformers 4.46.3** (driving the raw `tokenizer.json` + reference
`build_prompt`, bypassing the v5 tokenizer), split across the 6 GB GPU + CPU
(`device_map="auto"`, bf16, sdpa), generates with the production sampling params
(temp 0.6 / top_p 0.9 / top_k 50), and decodes via the validated
`ids_to_codes → quantizer.from_codes → Vocos` path.

Results:
- **Loads and runs** on the RTX 3050 6 GB (weights auto-split GPU+CPU). ~2–3 tok/s.
- **Deterministic:** same seed → byte-identical generated token sequence (run1==run2).
- **Pipeline correct** on the model's own documented example (`"नमस्ते।"`, speaker
  Amit): emits `<|end_of_speech|>` at ~85 tokens (~1.0 s), clean non-repetitive audio
  (unique-c0 frac = 1.00) → `artifacts/recon/11_baseline_amit_hindi_short.wav`.
- **Base-model Marathi-STEM gap observed:** the Marathi STEM sentence
  (`"पाणी हे हायड्रोजन आणि ऑक्सिजन यांचे संयुग आहे."`, speaker Anagha) did **not** emit
  `<|end_of_speech|>` within 700 tokens (over-generated to ~8.5 s) →
  `artifacts/recon/10_baseline_anagha_marathi.wav`. Same pipeline that stops cleanly on
  Hindi — so this is the documented base-model weakness on Marathi/"unusual words",
  i.e. exactly what this domain-adaptation project targets. Useful baseline signal.

Notes:
- **GPU CUDA wedge (resolved).** During the first heavy GPU allocation the CUDA runtime
  failed with `cudaErrorDevicesUnavailable` and wedged `nvidia_uvm` — subsequently every
  `torch.cuda` init failed with "CUDA unknown error" (while `nvidia-smi`/NVML kept
  working, since it's a different layer). A **reboot** cleared it; nothing is wrong with
  the GPU. Lesson: an unclean CUDA-process exit can wedge the driver until a uvm reload
  (`sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm`) or reboot.
- The parity *test proper* (`tests/test_reference_parity.py`, our `generate.py` vs the
  reference) is a Phase-5 artifact — `generate.py` doesn't exist yet. This step captured
  the reference baseline and proved the LM runs faithfully and deterministically here.

## 7. New decisions arising in Phase 0 (fold into DECISIONS.md once you confirm)

- **D11 (SNAC assertion).** Adopt the corrected contract check (conflict A). Alt
  considered: keep GOAL step-4 verbatim and let it fail (rejected — checks nothing).
- **D12 (reference provenance).** Vendored `bodhan-ai/indic-speak/inference.py` +
  `token_contract.md` into `reference/` as authoritative (you directed implementing
  here; no separate file was provided). Do-not-edit.
- **D13 (round-trip clip).** English LibriSpeech clip for codec validation; Marathi
  clip deferred to Phase-2 dataset selection. Alt: block Phase 0 on a Marathi clip
  (rejected — codec is language-agnostic; datasets loader is broken, E2).
- **Still open (on your ask-first list, not yet decided):** Marathi ASR (Phase 6),
  replay source (Phase 3/4), Marathi STEM training dataset (Phase 2). Stubs only.
