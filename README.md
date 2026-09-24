# Modhak-TTS — Marathi domain adaptation for Indic-Speak

A LoRA fine-tuning pipeline that domain-adapts **bodhan-ai/indic-speak** (closed-voice AR codec-LM TTS:
Llama-3.2-3B + SNAC 24 kHz + fine-tuned Vocos) toward Marathi, using only its existing named voices —
no new voices, no cloning, no style.

> **Built with Indic-Speak from Bodhan AI / AI4Bharat.**

## Status: complete — trained, evaluated, released

67 tests pass; ruff + mypy clean. The adapter was trained on an A100 (per-token loss 4.25→3.15) and
evaluated base-vs-fine-tuned with audio A/B samples. Released (public):
**[huggingface.co/BNarayanaReddy/modhak-tts-mr-lora](https://huggingface.co/BNarayanaReddy/modhak-tts-mr-lora)**.

> **Honest headline:** the fine-tune did **not** improve Marathi ASR-WER — it regressed, worst on the
> studio **anchor voice** (0.167→0.630), which is the *exact* drift the decision log pre-registered
> (D7/D9/D15); Hindi retention held. This is submitted as an honest domain-adaptation experiment with a
> negative result, not a quality upgrade — see **[docs/RESULTS.md](docs/RESULTS.md)**.

## How to read this repo

Suggested order for a reviewer:

1. **[docs/RESULTS.md](docs/RESULTS.md)** — what happened: the base-vs-fine-tuned table, the audible
   anchor drift, and the caveats.
2. **[DECISIONS.md](DECISIONS.md)** — the judgment log (D1–D18): every choice, its alternatives, why,
   and the runtime conflicts found and reported (this is the heart of the deliverable).
3. **[docs/OVERVIEW.md](docs/OVERVIEW.md)** — how the code is structured, phase by phase, with a module map.
4. **[docs/SETUP.md](docs/SETUP.md)** — run it yourself: env → weights → train → eval → sample.

## Repo map

```text
src/indic_speak_ft/   the pipeline package (tokens, data, model, train, inference, eval)
scripts/              entry points: recon_parity · validate_loader · build_eval_buckets · train · sample · eval
configs/              base/data/lora/train/eval yaml + eval_manifests/ (8 locked eval buckets)
reference/            authoritative, DO-NOT-EDIT: inference.py, token_contract.md, voices.md
docs/                 OVERVIEW · RESULTS · SETUP · DATA · TRAIN · EVAL · RECON
tests/                pytest (hermetic, < 2 min on CPU)
artifacts/            checkpoints/ (adapter) · eval_reports/ (panels) · samples/ (A/B wavs) · recon/
DECISIONS.md          the judgment log · SAFETY.md  the use/license policy
```

Per-phase detail and the full module table are in **[docs/OVERVIEW.md](docs/OVERVIEW.md)**.

## Quickstart

Full runbook (env, weights, gated ASR, server notes) is **[docs/SETUP.md](docs/SETUP.md)**. In short:

```bash
pip install -e ".[dev]"           # + ".[eval]" and torchaudio (matching CUDA) for the eval
python -m pytest -q               # 67 hermetic tests
python scripts/train.py           # LoRA run (server, full_gpu) -> artifacts/checkpoints/main_run/
python scripts/eval.py --label finetuned --adapter artifacts/checkpoints/main_run --with-speaker
python scripts/sample.py --text "…" --speaker Anagha --adapter artifacts/checkpoints/main_run
```

## Docs

| doc | contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | training + eval outcome, base-vs-fine-tuned table, audio A/B, caveats |
| [DECISIONS.md](DECISIONS.md) | judgment log D1–D18 (choices, alternatives, conflicts) |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | phase-by-phase build + module map |
| [docs/SETUP.md](docs/SETUP.md) | from-scratch runbook (laptop + server) |
| [docs/DATA.md](docs/DATA.md) · [docs/TRAIN.md](docs/TRAIN.md) · [docs/EVAL.md](docs/EVAL.md) | data pipeline · training recipe · eval methodology |
| [docs/RECON.md](docs/RECON.md) | Phase-0 findings + runtime conflicts |
| [SAFETY.md](SAFETY.md) | speaker-use policy, license, attribution |

## License & attribution

Derivative of `bodhan-ai/indic-speak` under the **Bodhan AI / Indic Open Model License**; distributed
under the same terms, with the required attribution above. Closed voice library only — no impersonation
or misleading content. Full policy + dataset licenses: **[SAFETY.md](SAFETY.md)**.
