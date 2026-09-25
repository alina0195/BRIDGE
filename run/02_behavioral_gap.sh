#!/usr/bin/env bash
# Table "gap": zero-shot, zero-shot + instruction, and the five objectives.
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh
BACKBONES="${BACKBONES:-llama mistral qwen}"

for bb in $BACKBONES; do
  backbone "$bb"
  evaluate "outputs/eval/$bb/zero_shot" --base-model "$BB_MODEL_ID"
  evaluate "outputs/eval/$bb/zero_shot_instructed" --base-model "$BB_MODEL_ID" \
    --instructed --tests 1,2,3
  for obj in $OBJECTIVES; do
    evaluate "outputs/eval/$bb/$obj" --base-model "$BB_MODEL_ID" \
      --adapter-path "$(adapter "$bb" "$obj")"
  done
done
python scripts/make_tables.py
