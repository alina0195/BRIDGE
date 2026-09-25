#!/usr/bin/env bash
# Retrieval-framework baselines: FLARE on each backbone, Self-RAG 7B.
#
# Protocol (bridge/baselines/protocol.py): test 1 and the test-4 vacuum run
# without retrieval, so SR/SDR and VRR are measured on the bare question like
# every other row; each question is decoded once.  --oracle-t1 adds test 1
# with gold-passage retrieval as a separate file that run_metrics ignores.
#
# Outputs: outputs/baselines/{llama,mistral,qwen}/flare, outputs/baselines/llama/selfrag
#
# Environment
#   PY           python for FLARE (needs nltk)                [python]
#   SELFRAG_PY   python for Self-RAG (needs vllm)              [$PY]
#   SELFRAG_TP   tensor-parallel size for Self-RAG             [number of visible GPUs]
#   BACKBONES    FLARE backbones                               [llama mistral qwen]
set -euo pipefail
cd "$(dirname "$0")/.."
. run/common.sh

PY="${PY:-python}"
SELFRAG_PY="${SELFRAG_PY:-$PY}"
SELFRAG_TP="${SELFRAG_TP:-$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | grep -c .)}"
BACKBONES="${BACKBONES:-llama mistral qwen}"

for bb in $BACKBONES; do
  backbone "$bb"
  out="outputs/baselines/$bb/flare"
  if done_eval "$out"; then echo "[skip] $out"; continue; fi
  "$PY" scripts/baselines/run_flare.py --model "$BB_MODEL_ID" \
    --eval-dir "$EVAL_DIR" --output-dir "$out" --oracle-t1
done

out="outputs/baselines/llama/selfrag"
if done_eval "$out"; then
  echo "[skip] $out"
else
  "$SELFRAG_PY" scripts/baselines/run_selfrag.py --model selfrag/selfrag_llama2_7b \
    --eval-dir "$EVAL_DIR" --output-dir "$out" \
    --tensor-parallel-size "$SELFRAG_TP" --dtype half --oracle-t1
fi
