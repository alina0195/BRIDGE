# Per-backbone settings shared by all drivers.  Source it, then call
#     backbone llama|mistral|qwen
#
# BB_LAYER       hook layer: the last block of each backbone
# BB_ALPHA       BRIDGe injection magnitude.  Llama and Mistral use 20
#                (k = 2.58 and 4.57 class gaps); on Qwen the class gap is 65.4,
#                so alpha = 20 is only k = 0.31 and we match Llama's k = 2.58
# BB_TAU_C       presence threshold of the BRIDGe dual gate
# BB_TAU_C_PRES  presence threshold tuned for the presence-only baseline
# BB_TAU_C_GRID  presence thresholds swept for BRIDGe (a ladder around the
#                midpoint of the no-context / grounded presence projections)
# TAU_V_RATES    tau_v is set by quantile: the value at which this share of
#                stable prompts passes the volatility condition

TAU_V_RATES="0.40 0.30 0.20"
TAU_V_RATE="0.30"

backbone() {
  case "${1:-}" in
    llama)
      BB_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
      BB_LAYER=31
      BB_ALPHA=20.0
      BB_TAU_C=0.5
      BB_TAU_C_PRES=-0.5
      BB_TAU_C_GRID="-0.5 0.0 0.5"
      ;;
    mistral)
      BB_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
      BB_LAYER=31
      BB_ALPHA=20.0
      BB_TAU_C=0.5
      BB_TAU_C_PRES=0.68854532282371
      BB_TAU_C_GRID="-0.5 0.0 0.5"
      ;;
    qwen)
      BB_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
      BB_LAYER=27
      BB_ALPHA=168.517
      BB_TAU_C=19.16
      BB_TAU_C_PRES=19.105009002660715
      BB_TAU_C_GRID="-0.725 4.32 11.688 19.16"
      ;;
    *)
      echo "unknown backbone '${1:-}' (llama | mistral | qwen)" >&2
      return 2
      ;;
  esac
  BB_KEY="$1"
  BB_DIRS="outputs/directions/${BB_KEY}"
  BB_PRESENCE="${BB_DIRS}/presence/layer_${BB_LAYER}_presence.npy"
  BB_VHAT="${BB_DIRS}/hook/layer_${BB_LAYER}_vhat.npy"
  BB_VHAT_META="${BB_DIRS}/hook/layer_${BB_LAYER}_vhat.json"
  export BB_KEY BB_MODEL_ID BB_LAYER BB_ALPHA BB_TAU_C BB_TAU_C_PRES BB_TAU_C_GRID \
         BB_DIRS BB_PRESENCE BB_VHAT BB_VHAT_META
}
