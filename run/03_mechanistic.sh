#!/usr/bin/env bash
# Mechanistic diagnosis (Section 3.2 and appendix figures), per backbone:
#   1. volatility probe on zero-shot and the five fine-tuned variants
#   2. activation patching on CA-AVMD
#   3. ablation targets from the patching run
#   4. targeted ablation + three random-component controls (seeds 0, 1, 2)
#   5. probe survival under the same ablation
# then the ablation table and the four figures.  Finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh
BACKBONES="${BACKBONES:-llama mistral qwen}"

PROBE_LAYERS="-1,-4,-8,-16"
PATCH_TOP_K=6
ABLATION_LIMIT=150
SURVIVAL_LIMIT=200
N_MEAN=100
RANDOM_SEEDS="0 1 2"

# Llama: the ablated components are the attention-dominant heads of L16 and
# the MLPs of L17, L26 and L29 located by a 100-pair patching run with
# --min-gap 0.05, plus the two strongest heads of L17.  Set LLAMA_TARGETS=""
# to derive them from the patching run of step 2 instead.
if [ -z "${LLAMA_TARGETS+x}" ]; then
  LLAMA_TARGETS="--heads 16:9,16:18,16:25,17:4,17:5 --mlps 17,26,29"
fi

have() { [ -f "$1" ] && echo "[skip] $1"; }

for bb in $BACKBONES; do
  backbone "$bb"
  mech="outputs/mechanistic/$bb"
  ckpt="$(adapter "$bb" ca_avmd)"

  # 1. Volatility probe per variant.
  for variant in zero_shot $OBJECTIVES; do
    out="$mech/probes/$variant"
    have "$out/probe_summary.json" && continue
    extra=()
    [ "$variant" = zero_shot ] || extra=(--adapter-path "$(adapter "$bb" "$variant")")
    python scripts/mechanistic/volatility_probe.py --model "$BB_MODEL_ID" ${extra[@]+"${extra[@]}"} \
      --eval-dir "$EVAL_DIR" --target-layers="$PROBE_LAYERS" --output-dir "$out"
  done

  # Per-backbone settings: patching pairs and minimum P(defer) gap; the
  # random controls draw their layers from [8, rand_max].
  case "$bb" in
    llama)   n_pairs=50;  min_gap=0.02;  rand_max=28 ;;
    mistral) n_pairs=100; min_gap=0.005; rand_max=31 ;;
    qwen)    n_pairs=100; min_gap=0.005; rand_max=27 ;;
  esac

  # 2. Activation patching on CA-AVMD.
  patch="$mech/patching/results.json"
  have "$patch" || python scripts/mechanistic/activation_patching.py \
    --model "$BB_MODEL_ID" --adapter-path "$ckpt" --eval-dir "$EVAL_DIR" \
    --output-dir "$mech/patching" --n-pairs "$n_pairs" --top-k-layers "$PATCH_TOP_K" \
    --min-gap "$min_gap"

  # 3. Ablation targets.
  if [ "$bb" = llama ] && [ -n "$LLAMA_TARGETS" ]; then
    targets="$LLAMA_TARGETS"
  else
    targets="$(python scripts/mechanistic/extract_ablation_targets.py "$patch")"
  fi
  mkdir -p "$mech/ablation"
  echo "$targets" > "$mech/ablation/target_flags.txt"
  echo "[$bb] ablation targets: $targets"

  # 4. Targeted ablation and random-component controls.
  ablate=(--model "$BB_MODEL_ID" --adapter-path "$ckpt" --eval-dir "$EVAL_DIR"
          --limit "$ABLATION_LIMIT" --n-mean-samples "$N_MEAN" --ablation-mode mean)
  # shellcheck disable=SC2086
  have "$mech/ablation/targets/results.json" || python scripts/mechanistic/targeted_ablation.py \
    "${ablate[@]}" $targets --output-dir "$mech/ablation/targets"
  for seed in $RANDOM_SEEDS; do
    out="$mech/ablation/random_seed$seed"
    # shellcheck disable=SC2086
    have "$out/results.json" || python scripts/mechanistic/targeted_ablation.py \
      "${ablate[@]}" $targets --random-control --random-seed "$seed" \
      --random-layer-min 8 --random-layer-max "$rand_max" --output-dir "$out"
  done

  # 5. Probe survival.
  # shellcheck disable=SC2086
  have "$mech/probe_survival/results.json" || python scripts/mechanistic/probe_survival.py \
    --model "$BB_MODEL_ID" --adapter-path "$ckpt" --eval-dir "$EVAL_DIR" \
    --limit "$SURVIVAL_LIMIT" --n-mean-samples "$N_MEAN" --ablation-mode mean \
    $targets --output-dir "$mech/probe_survival"
done

python scripts/plots/plot_probe_cosine.py --backbones "$BACKBONES"

# The table and the other three figures combine all three backbones.
for bb in llama mistral qwen; do
  for f in patching/results.json ablation/targets/results.json \
           ablation/random_seed2/results.json probe_survival/results.json; do
    [ -f "outputs/mechanistic/$bb/$f" ] || { echo "[wait] outputs/mechanistic/$bb/$f"; exit 0; }
  done
done
python scripts/mechanistic/build_ablation_table.py
python scripts/plots/plot_patching_overlay.py
python scripts/plots/plot_targeted_ablation.py
python scripts/plots/plot_probe_survival.py
