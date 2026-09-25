#!/usr/bin/env bash
# Ablations of Section 5, all on the CA-AVMD checkpoint:
#   direction    ungated steering along -c_hat, +v_hat, random, shuffled-label v_hat
#   gates        the BRIDGe injection under each gate condition
#   sensitivity  presence-only gate: alpha, tau_c and hook layer (Llama)
#   seeds        three training seeds, BRIDGe with frozen directions and thresholds
# Run 04_bridge.sh first (directions and thresholds).
#   PARTS="direction gates" bash run/05_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh
BACKBONES="${BACKBONES:-llama mistral qwen}"
PARTS="${PARTS:-direction gates sensitivity seeds}"
K_LIST="${K_LIST:-1 2 4}"
SEEDS="${SEEDS:-42 123 7}"
SEED_BACKBONES="${SEED_BACKBONES:-llama mistral}"
want() { case " $PARTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# steer_args: sets the array SA to the CA-AVMD checkpoint and hook of $BB_KEY.
steer_args() {
  SA=(--base-model "$BB_MODEL_ID" --adapter-path "$(adapter "$BB_KEY" ca_avmd)"
      --layer "$BB_LAYER" --presence-dir "$BB_PRESENCE" --volatile-dir "$BB_VHAT")
}

if want direction; then
  for bb in $BACKBONES; do
    backbone "$bb"; steer_args
    shuf="$BB_DIRS/hook/layer_${BB_LAYER}_vhat_shuf0.npy"
    [ -f "$shuf" ] || python scripts/fit_hook_direction.py --layer "$BB_LAYER" \
      --shuffle-labels --fit-seed 0 --output-dir "$BB_DIRS/hook"
    for k in $K_LIST; do
      for d in presence volatility random shuffled; do
        case "$d" in
          presence)   src=(--steer presence) ;;
          volatility) src=(--steer volatility --steer-dir "$BB_VHAT") ;;
          random)     src=(--steer random --steer-seed 0) ;;
          shuffled)   src=(--steer volatility --steer-dir "$shuf") ;;
        esac
        evaluate "outputs/ablations/direction/$bb/${d}_k${k}" "${SA[@]}" \
          --gate none --dir-meta "$BB_VHAT_META" --k "$k" "${src[@]}"
      done
    done
  done
fi

if want gates; then
  for bb in $BACKBONES; do
    backbone "$bb"; steer_args
    tv="$(tau_v_for "$TAU_V_RATE")"
    for gate in none volatile_only presence_only full; do
      evaluate "outputs/ablations/gates/$bb/$gate" "${SA[@]}" --alpha "$BB_ALPHA" \
        --gate "$gate" --tau-v "$tv" --tau-c "$BB_TAU_C"
    done
  done
fi

if want sensitivity; then
  backbone llama
  ckpt="$(adapter llama ca_avmd)"
  for l in 16 24; do
    [ -f "$BB_DIRS/presence/layer_${l}_presence.npy" ] || python scripts/fit_presence_probe.py \
      --model "$BB_MODEL_ID" --adapter-path "$ckpt" --layers 16,24 \
      --output-dir "$BB_DIRS/presence"
  done
  base=(--base-model "$BB_MODEL_ID" --adapter-path "$ckpt" --gate presence_only)
  root="outputs/ablations/sensitivity/llama"
  for a in 10 20 25 50 75 100; do
    evaluate "$root/alpha_${a}" "${base[@]}" --layer 31 --presence-dir "$BB_PRESENCE" \
      --alpha "$a" --tau-c -0.5
  done
  for tc in -0.7 -0.6 -0.4 -0.3; do
    evaluate "$root/tau_c_${tc}" "${base[@]}" --layer 31 --presence-dir "$BB_PRESENCE" \
      --alpha 50 --tau-c "$tc"
  done
  for l in 16 24; do
    # same overall fire rate as the layer-31 gate (alpha 50, tau_c -0.5)
    evaluate "$root/layer_${l}_matched" "${base[@]}" --layer "$l" \
      --presence-dir "$BB_DIRS/presence/layer_${l}_presence.npy" --alpha 50 \
      --match-fire-rate 0.338
  done
  evaluate "$root/layer_16_fixed" "${base[@]}" --layer 16 \
    --presence-dir "$BB_DIRS/presence/layer_16_presence.npy" --alpha 50 --tau-c -0.5
fi

if want seeds; then
  for bb in $SEED_BACKBONES; do
    backbone "$bb"
    tv="$(tau_v_for "$TAU_V_RATE")"
    for seed in $SEEDS; do
      for obj in $OBJECTIVES; do
        model="outputs/models/seeds/${bb}_${obj}_seed${seed}"
        [ -d "$model/adapter" ] || PYTHONHASHSEED="$seed" python scripts/train.py \
          --base-model "$BB_MODEL_ID" --objective "$obj" --data "$EVAL_DIR/train.jsonl" \
          --seed "$seed" --output-dir "$model"
        evaluate "outputs/seeds/$bb/${obj}_seed${seed}" --base-model "$BB_MODEL_ID" \
          --adapter-path "$model/adapter"
      done
      # BRIDGe with the directions and thresholds of the main run, frozen
      evaluate "outputs/seeds/$bb/bridge_seed${seed}" --base-model "$BB_MODEL_ID" \
        --adapter-path "outputs/models/seeds/${bb}_ca_avmd_seed${seed}/adapter" \
        --layer "$BB_LAYER" --presence-dir "$BB_PRESENCE" --volatile-dir "$BB_VHAT" \
        --alpha "$BB_ALPHA" --gate full --tau-v "$tv" --tau-c "$BB_TAU_C"
    done
  done
fi
python scripts/make_tables.py
