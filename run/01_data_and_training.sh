#!/usr/bin/env bash
# Data splits and the five training objectives on every backbone (seed 42).
#   BACKBONES="llama" OBJECTIVES="ca_avmd" bash run/01_data_and_training.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh
BACKBONES="${BACKBONES:-llama mistral qwen}"
SEED="${SEED:-42}"

[ -f "$EVAL_DIR/train.jsonl" ] || python scripts/prepare_data.py --output-dir "$EVAL_DIR"

for bb in $BACKBONES; do
  backbone "$bb"
  for obj in $OBJECTIVES; do
    out="outputs/models/${bb}_${obj}"
    if [ -d "$out/adapter" ]; then echo "[skip] $out"; continue; fi
    PYTHONHASHSEED="$SEED" python scripts/train.py --base-model "$BB_MODEL_ID" \
      --objective "$obj" --data "$EVAL_DIR/train.jsonl" --seed "$SEED" --output-dir "$out"
  done
done
