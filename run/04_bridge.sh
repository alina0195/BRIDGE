#!/usr/bin/env bash
# BRIDGe on the CA-AVMD checkpoint of each backbone (Table "main results").
#   1. presence probe c_hat and v_hat at the hook layer
#   2. presence-only baseline at its tuned tau_c
#   3. dual-gate grid: tau_v at 40/30/20% stable pass x the tau_c ladder
# The reported operating point is tau_v at 30% stable pass and BB_TAU_C, linked
# as outputs/bridge/<bb>/bridge; make_tables.py also lists the whole grid.
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh
BACKBONES="${BACKBONES:-llama mistral qwen}"

for bb in $BACKBONES; do
  backbone "$bb"
  ckpt="$(adapter "$bb" ca_avmd)"
  [ -f "$BB_PRESENCE" ] || python scripts/fit_presence_probe.py --model "$BB_MODEL_ID" \
    --adapter-path "$ckpt" --layers "$BB_LAYER" --output-dir "$BB_DIRS/presence"
  [ -f "$BB_VHAT" ] || python scripts/fit_hook_direction.py --base-model "$BB_MODEL_ID" \
    --adapter-path "$ckpt" --layer "$BB_LAYER" --output-dir "$BB_DIRS/hook"

  steer=(--base-model "$BB_MODEL_ID" --adapter-path "$ckpt" --layer "$BB_LAYER"
         --presence-dir "$BB_PRESENCE" --volatile-dir "$BB_VHAT" --alpha "$BB_ALPHA")

  evaluate "outputs/bridge/$bb/presence_only" "${steer[@]}" \
    --gate presence_only --tau-c "$BB_TAU_C_PRES"

  for rate in $TAU_V_RATES; do
    tv="$(tau_v_for "$rate")"
    for tc in $BB_TAU_C_GRID; do
      evaluate "outputs/bridge/$bb/grid/tv${rate}_tc${tc}" "${steer[@]}" \
        --gate full --tau-v "$tv" --tau-c "$tc"
    done
  done
  ln -sfn "grid/tv${TAU_V_RATE}_tc${BB_TAU_C}" "outputs/bridge/$bb/bridge"
done
python scripts/make_tables.py
