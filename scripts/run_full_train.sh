#!/usr/bin/env bash
# Server launcher for the full LoRA run: sets the HF token (gated Rasa) and runs training.
set -euo pipefail
cd /root/Modhak-TTS
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u scripts/train.py
