# Modhak-TTS — Marathi domain adaptation for Indic-Speak

LoRA fine-tuning pipeline that domain-adapts **bodhan-ai/indic-speak** (a closed-voice
AR codec-LM TTS model, Llama-3.2-3B backbone + SNAC 24 kHz + fine-tuned Vocos) toward
Marathi, using the existing named voices (no new voices, no voice cloning, no style).

> **Built with Indic-Speak from Bodhan AI / AI4Bharat.**

## Status: complete — trained + evaluated on A100

**67 tests pass; ruff + mypy clean.** The LoRA adapter was trained on the server (loss 4.25→3.15)
and evaluated base-vs-fine-tuned across the locked buckets. **Honest headline: the fine-tune did not
improve ASR-WER and regressed the anchor voice — the exact drift Decisions 7/9/D15 pre-registered;
Hindi retention held.** Full numbers, caveats, and interpretation: **[docs/RESULTS.md](docs/RESULTS.md)**.

Setup/repro: **[docs/SETUP.md](docs/SETUP.md)**; judgment log: **[DECISIONS.md](DECISIONS.md)** (D1–18);
recipe **[docs/TRAIN.md](docs/TRAIN.md)**, eval **[docs/EVAL.md](docs/EVAL.md)**, policy
**[SAFETY.md](SAFETY.md)**.

- **Phase 0 (recon):** ✅ token contract resolved live (`configs/tokens_resolved.yaml`),
  **SNAC↔Vocos round-trip** (mel-corr 0.973), **reference baseline + determinism** pass.
  Findings/conflicts in **[docs/RECON.md](docs/RECON.md)**.
- **Phase 1 (token contract):** ✅ `tokens.py` — frame math + byte-for-byte interleave parity.
- **Phase 2 (data):** ✅ schema, gates, D15 speaker map, Marathi text-norm, mixture, collator,
  and **8 locked eval buckets** (`configs/eval_manifests/`, 220 clips held out). Loader
  validated on the real corpora; SPRINGLab per-speaker quality checked. See
  **[docs/DATA.md](docs/DATA.md)**.
- **Phase 3 (model):** ✅ LoRA (Decision 2), freeze policy + verifier — **21.43 M trainable
  (0.649%)**.
- **Phase 4 (training):** ✅ SNAC compile, per-codebook weighted loss (D4+D14), replay,
  regression-stop, determinism, trainer, `scripts/train.py --smoke`. Runs on the **server**
  (the 11 GB laptop OOMs under bf16+offload). Recipe: **[docs/TRAIN.md](docs/TRAIN.md)**.
- **Phase 5 (inference):** ✅ `generate.py` (reference semantics + adapter attach/merge),
  `decode.py` (SNAC/Vocos), `scripts/sample.py` — reference parity tested (de-interleave +
  prompt byte-for-byte; full e2e gated to the server).
- **Phase 6 (eval):** ✅ WER (Bodhan `indic-transcribe-core`, ins/del/sub), calibrated speaker
  similarity, prosody, retention, longform, stock-vs-Vocos A/B, panel → **HTML report**
  (`scripts/eval.py`). Metric core unit-tested. See **[docs/EVAL.md](docs/EVAL.md)**.

**Done:** full `train.py` on the A100 (adapter in `artifacts/checkpoints/main_run/`, weights gitignored)
→ `eval.py` base-vs-fine-tuned → reports in `artifacts/eval_reports/{baseline,finetuned}/panel.{json,html}`
→ results written up in **[docs/RESULTS.md](docs/RESULTS.md)**. The laptop handles inference/sampling;
training + the ASR-download eval ran on the server.

## Layout

```text
reference/      authoritative, do-not-edit: inference.py, token_contract.md, voices.md
configs/        tokens_resolved.yaml (P0); hydra configs added per phase
docs/           RECON.md (P0); DATA/TRAIN/EVAL added per phase
src/indic_speak_ft/   pipeline package (built per phase)
tests/          pytest (smoke tests < 2 min CPU)
scripts/        prepare_data / train / eval / sample entry points
artifacts/      gitignored except artifacts/recon/ (P0 audio evidence)
```

## Environment

- Inference on this laptop (RTX 3050 6 GB). Training on a separate server.
- Model weights + codec live on the external offload drive, never in git.
- Requires a venv pinned to `transformers==5.14.1` (the model's save version) and
  `datasets>=3` — see RECON §1 (E1–E3). `pip install -e .` then `pip install -e .[dev]`.

## Reproduce (once Phase 1+ lands)

```bash
scripts/train.py --smoke     # 100 steps / 20 samples, proves the pipeline
scripts/train.py             # main run (server)
scripts/eval.py              # metric vector + HTML report
scripts/sample.py            # synthesize sample wavs
```

## License / attribution

Indic-Speak is under the Bodhan AI / Indic Open Model License (see `reference/` and,
when written, `SAFETY.md`). Attribution string above is required. Closed voice library
only; no unauthorized impersonation or misleading content.

```bibtex
@misc{indicspeak2026,
  title  = {Indic-Speak: Text-to-Speech for 22 Indian Languages and English},
  author = {Bodhan AI and AI4Bharat},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-speak}
}
```
