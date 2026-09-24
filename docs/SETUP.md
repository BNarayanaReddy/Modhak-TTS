# SETUP — from-scratch runbook (laptop + server)

Reproducible setup for the two environments this project runs in:

- **Laptop** (RTX 3050 6 GB) — recon + inference/eval; the LM loads split GPU+CPU.
- **Server** (bigger GPU) — training / heavy runs; the LM loads fully on GPU.

This runbook is updated every phase with the commands that phase adds. Commands are
copy-pasteable; adjust the two paths (`REPO`, `MODELS`) to the box.

```bash
export REPO=/path/to/Modhak-TTS                 # this repo
export MODELS=/path/to/models                   # where weights live (NOT in git)
#   laptop today: REPO=/home/narayana/ai/bodhan/Modhak-TTS
#   laptop today: MODELS=/media/narayana/Windows-SSD/ai-models/bodhan_offload/models
```

---

## 1. Prerequisites

- Python **3.10** (the pins target 3.10; `python3.10 --version`).
- A CUDA GPU + recent NVIDIA driver. Check the runtime actually works (NVML alone is
  not enough — see the wedge note at the bottom):
  ```bash
  nvidia-smi
  python3 -c "import torch; print(torch.cuda.is_available()); torch.ones(1,device='cuda')"
  ```
- `git`, and a Hugging Face account/token (the model is public but auth avoids rate limits).

## 2. Get the code

```bash
git clone <your-remote>/Modhak-TTS "$REPO" && cd "$REPO"    # or copy the directory
```

## 3. Python environment

**Server (clean, recommended)** — a fresh venv with the pinned deps, incl.
`transformers==5.14.1` (the version the model was saved with; loads the tokenizer
natively):

```bash
cd "$REPO"
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Install torch matching THIS box's CUDA (pyproject leaves torch unpinned on purpose).
# Example — pick the index-url for the server's CUDA (cu121/cu124/cu128/…):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"        # transformers 5.14.1, snac, accelerate, peft, datasets>=3, pytest, …
# For the eval run (Phase 6) also install the eval extras. torchaudio must come from the SAME
# CUDA index-url as torch (plain PyPI pulls a CPU/mismatched build):
pip install torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[eval]"       # jiwer, sentencepiece, speechbrain, hyperpyyaml, ruamel.yaml<0.19
```

> The A100 server here used **uv** to manage the venv (`uv venv` + `uv pip install --python
> .venv/bin/python <pkg>`; no `pip` inside the venv). Same package set — substitute `uv pip install`
> for `pip install` above. The **eval ASR** (`bodhan-ai/indic-transcribe-core`) and the ECAPA
> embedder are **gated / downloaded at first run** — accept the ASR terms on HF and `hf auth login`
> (or `export HF_TOKEN=…`) before `eval.py`. The ASR is ~2.5 GB (the `nemo/` checkpoint is skipped).

**Laptop (as used for recon)** — the box already has `torch 2.13.0+cu130` and
`transformers 4.46.3` in `~/.local`; the venv inherits them to avoid a 2.5 GB torch
reinstall:

```bash
uv venv "$REPO/.venv" --python 3.10 --system-site-packages
# recon runs on the inherited transformers 4.46.3 (see note E1) via a raw-tokenizer shim;
# for a clean pinned env do the server steps instead.
```

> **Why transformers 5.14.1:** the tokenizer was saved with a v5-only
> `TokenizersBackend` class; transformers 4.x cannot `AutoTokenizer.from_pretrained` it.
> Recon on the laptop sidesteps this by reading `tokenizer.json` directly. On the server,
> just install 5.14.1 and it loads natively. (See docs/RECON.md E1.)

## 4. Hugging Face auth

```bash
huggingface-cli login            # or: export HF_TOKEN=hf_xxx
```

## 5. Download weights (LM + SNAC codec + fine-tuned Vocos)

Server has room for the full snapshot; download once into `$MODELS`:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
md = os.environ["MODELS"]
# Full indic-speak repo (model.safetensors 6.6 GB, tokenizer, config, vocos/best.pt, …)
snapshot_download("bodhan-ai/indic-speak", local_dir=f"{md}/indic-speak")
# SNAC 24 kHz codec
for f in ("config.json", "pytorch_model.bin"):
    hf_hub_download("hubertsiuzdak/snac_24khz", f, local_dir=f"{md}/snac_24khz")
print("done ->", md)
PY
```

On the **laptop** these were fetched piecemeal to the NTFS offload drive with
`local_dir` (no symlinks); the SNAC + Vocos are ~550 MB, the LM 6.6 GB.

## 6. Verify (recon)

Reproduce the Phase-0 checks. `recon_parity.py` expects `$MODELS/indic-speak` and
`$MODELS/snac_24khz` (it takes `--offload` = the dir that *contains* `models/`, so pass
the parent):

```bash
# token-contract + interleave + SNAC-band tests (hermetic, <2s):
python -m pytest tests/test_tokens.py -q
# also validate the contract against the real vocab:
INDIC_SPEAK_TOKENIZER_JSON="$MODELS/indic-speak/tokenizer.json" \
  python -m pytest tests/test_tokens.py::test_real_tokenizer_contract_if_available -q

# reference baseline + determinism (short, ~1 s clip). --offload is the parent of models/:
python scripts/recon_parity.py --offload "$(dirname "$MODELS")" \
  --text "नमस्ते।" --speaker Amit --runs 1 --max-new-tokens 250 --gpu-mem 5GiB
#   server with a big GPU: raise --gpu-mem (e.g. 40GiB) so nothing offloads to CPU.
```

Expected: `end_of_speech emitted=True`, `unique-c0 frac=1.00`, a ~1 s wav in
`artifacts/recon/`. The SNAC↔Vocos round-trip evidence is already in `artifacts/recon/`
(`03_nodedup_vocos.wav` ≈ original).

## 7. Per-phase commands

| Phase | Command | Status |
|---|---|---|
| 1 — token contract | `python -m pytest tests/test_tokens.py -q` | done |
| 2 — data pipeline | `python scripts/validate_loader.py` · `python scripts/build_eval_buckets.py` | done (buckets locked) |
| 3 — model/LoRA/freeze | `python -m pytest tests/test_freeze_policy.py -q` | done |
| 4 — train (smoke) | `python scripts/train.py --smoke` | ran on server |
| 4 — train (full) | `python scripts/train.py`  (set `full_gpu: true`) | **ran** → `artifacts/checkpoints/main_run/` |
| 5 — inference | `python scripts/sample.py --text "…" --speaker Anagha [--adapter <dir>]` | done |
| 6 — eval | `python scripts/eval.py --label finetuned --adapter <dir>` | **ran** → `artifacts/eval_reports/`; see [RESULTS.md](RESULTS.md) |

Run the whole hermetic suite with `python -m pytest -q` (67 tests; set
`INDIC_SPEAK_TOKENIZER_JSON` and `SNAC_DIR` to also run the real-asset tests).

### Training on the server (Phase 4)

Training is RAM-heavy and belongs on the server (the laptop's 11 GB OOMs under bf16 + CPU
offload). On the server:

```bash
# in configs/base.yaml set:  full_gpu: true   (load the base entirely on one GPU, no offload)
python scripts/train.py --smoke     # 100 steps / 20 samples — proves the pipeline end-to-end
python scripts/train.py             # full run
```

Adapters + config snapshots land under `artifacts/checkpoints/{smoke_run,main_run}/`. See docs/TRAIN.md.

### Eval on the server (Phase 6)

Base-vs-fine-tuned over the 8 locked buckets → `panel.json` + `panel.html` per label. Needs the
`.[eval]` extras and HF auth for the gated ASR (above). Run on the GPU box (`device="cuda"` when
`full_gpu: true`). Synthesis is autoregressive (batch-1, latency-bound) so it is the slow step —
run it in `tmux`/`nohup`; **don't** run base and fine-tuned in parallel (two AR processes thrash
the GPU and each runs slower).

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)          # gated ASR + Rasa refs
# canonical full run (both labels, all buckets, with per-voice speaker similarity):
bash scripts/run_eval.sh                                    # = eval.py --label baseline … then --label finetuned …
# or one label at a time:
python scripts/eval.py --label baseline  --with-speaker
python scripts/eval.py --label finetuned --adapter artifacts/checkpoints/main_run --with-speaker
# fast WER-only variant (smaller n, lower token cap — what the deadline run used):
bash scripts/run_eval_fast.sh 6 1000                        # --limit 6 --max-new-tokens 1000, no speaker sim
```

Reports land under `artifacts/eval_reports/{baseline,finetuned}/panel.{json,html}` (+ saved
`eval_outputs/*_hyps.json`). Interpretation + the base-vs-fine-tuned table: **[RESULTS.md](RESULTS.md)**.
Flags: `--with-speaker` (adds ECAPA speaker similarity + the real reference-audio fetch), `--limit N`
(cap items/bucket), `--max-new-tokens N` (generation cap; 2520 ≈ 30 s is the configured default).

## Notes / gotchas

- **CUDA runtime vs NVML.** `nvidia-smi` (NVML) can work while the CUDA *runtime* is
  wedged. If `torch.ones(1, device='cuda')` throws `CUDA unknown error` after an unclean
  CUDA-process crash, reset the driver: `sudo modprobe -r nvidia_uvm && sudo modprobe
  nvidia_uvm` (or reboot). This bit us once in Phase 0 (docs/RECON.md §6).
- **6 GB VRAM** can't hold the 3.3B bf16 LM (~6.6 GB) plus KV cache; the laptop uses
  `device_map="auto"` + a GPU cap so a couple of layers sit on CPU. Faithful (bf16), just
  ~2–3 tok/s. Do **not** switch to 4-bit for parity — it changes logits.
- **datasets.** Pin `datasets>=3` (Phase 2). The laptop's global `datasets 2.14.4` is
  broken against the installed `pyarrow` (docs/RECON.md E2); the venv fixes it.
- **Never commit weights.** `.gitignore` excludes `*.safetensors|*.bin|*.pt`; weights
  live under `$MODELS` only.
