#!/usr/bin/env bash
# Server eval runner: base model then fine-tuned adapter, both with speaker similarity.
# Produces artifacts/eval_reports/{baseline,finetuned}/{panel.json,panel.html,eval_outputs/}.
set -euo pipefail
cd /root/Modhak-TTS
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export TOKENIZERS_PARALLELISM=false
P=.venv/bin/python
ADAPTER=artifacts/checkpoints/main_run

echo "===== BASELINE (base model) ====="
$P scripts/eval.py --label baseline --with-speaker "$@"
echo "===== FINETUNED (adapter=$ADAPTER) ====="
$P scripts/eval.py --label finetuned --adapter "$ADAPTER" --with-speaker "$@"
echo "ALL_EVAL_DONE"
