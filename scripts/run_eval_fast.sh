#!/usr/bin/env bash
# Fast WER-only pass: base then fine-tuned, no speaker similarity (no reference download / embed).
#   run_eval_fast.sh [limit] [max_new_tokens]
set -euo pipefail
cd /root/Modhak-TTS
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export TOKENIZERS_PARALLELISM=false
LIM=${1:-6}; MNT=${2:-1000}
P=.venv/bin/python
echo "===== BASELINE (WER-only, limit=$LIM mnt=$MNT) ====="
$P scripts/eval.py --label baseline --limit "$LIM" --max-new-tokens "$MNT"
echo "===== FINETUNED (WER-only, limit=$LIM mnt=$MNT) ====="
$P scripts/eval.py --label finetuned --adapter artifacts/checkpoints/main_run --limit "$LIM" --max-new-tokens "$MNT"
echo "ALL_EVAL_DONE"
