# Helpers shared by the run drivers.  Source from the BRIDGe root.
. configs/backbones.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EVAL_DIR="${EVAL_DIR:-data/evowiki}"
OBJECTIVES="${OBJECTIVES:-sft vmd avmd ca_vmd ca_avmd}"

# done_eval DIR: true when DIR holds a finished evaluation.
done_eval() { [ -f "$1/summary.json" ] && [ -f "$1/config.json" ]; }

# evaluate OUT_DIR ARGS...: run scripts/evaluate.py unless OUT_DIR is complete.
evaluate() {
  local out="$1"; shift
  if done_eval "$out"; then echo "[skip] $out"; return 0; fi
  python scripts/evaluate.py --eval-dir "$EVAL_DIR" --output-dir "$out" "$@"
}

adapter() { echo "outputs/models/${1}_${2}/adapter"; }

# tau_v_for RATE: tau_v at the given stable pass rate, 4 significant figures.
tau_v_for() {
  python -c "import json,sys; m=json.load(open(sys.argv[1])); \
print(f\"{m['tau_v_at_stable_pass'][f'{float(sys.argv[2]):.2f}']:.4g}\")" "$BB_VHAT_META" "$1"
}
