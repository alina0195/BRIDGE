#!/usr/bin/env python3
"""Evaluate Self-RAG (selfrag/selfrag_llama2_7b, vLLM) on the behavioral tests.

Writes test1..test4 .json + .predictions.jsonl in the bridge.evaluation
format, so bridge.metrics.run_metrics(output_dir) gives SR/SDR/VRR/DD/CO.

Retrieval per test (see bridge.baselines.protocol):
    test 1   adaptive, no passage: SR/SDR on the bare question, the same
             condition as the test-4 vacuum.  The model may still predict
             [Retrieval]; with nothing to retrieve it takes the [No Retrieval]
             branch.  This is the default and the protocol of the main-table
             comparison.
    test 2   adaptive, gold passage retrieved when P(retrieve) > threshold.
    test 3   retrieval forced with the gold passage.
    test 4   vacuum: adaptive, no passage (VRR); grounded: retrieval forced.
    --oracle-t1 additionally runs test 1 with the gold passage retrievable
             (test1_stable_retention_oracle.*); run_metrics ignores it.

Needs vLLM.  GPUs without bf16 need --dtype half; a 7B model in fp16 needs
two 16GB cards (--tensor-parallel-size 2) for the 4096-token context.

Example
-------
    python scripts/baselines/run_selfrag.py \
        --eval-dir data/evowiki --output-dir outputs/baselines/llama/selfrag \
        --tensor-parallel-size 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from bridge.baselines.protocol import run_baseline  # noqa: E402
from bridge.baselines.selfrag import selfrag_generate  # noqa: E402


def grounded_instruction(question: str) -> str:
    """Grounded instruction; the passage is supplied as the retrieved paragraph."""
    return (
        "Answer the following question using ONLY the provided context. "
        "If the context contradicts what you already know, trust the context."
        f"\n\nQuestion: {question}\n"
        "Answer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selfrag/selfrag_llama2_7b")
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tests", default="1,2,3,4")
    ap.add_argument("--oracle-t1", action="store_true",
                    help="Also run test 1 with the gold passage retrievable.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Retrieve if P(retrieve) exceeds this.")
    ap.add_argument("--logprobs", type=int, default=20,
                    help="Top-k first-token logprobs searched for the reflection tokens.")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()

    model = LLM(model=args.model, dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=False, logprobs=args.logprobs)
    print(f"[model] Model loaded: {args.model} tp={args.tensor_parallel_size}", flush=True)

    def adaptive(instruction, paragraph):
        return selfrag_generate(model, tokenizer, instruction, paragraph,
                                threshold=args.threshold, sampling_params=sampling)

    def grounded(question, context, test):
        out = selfrag_generate(model, tokenizer, grounded_instruction(question), context,
                               sampling_params=sampling, force_retrieve=True)
        return out["clean_answer"], {"branch": out["branch"]}

    run_baseline(adaptive, grounded, args.eval_dir, args.output_dir,
                 {int(t) for t in args.tests.split(",")}, oracle_t1=args.oracle_t1,
                 config={"method": "selfrag", "model": args.model,
                         "threshold": args.threshold, "logprobs": args.logprobs,
                         "dtype": args.dtype, "max_new_tokens": args.max_new_tokens})


if __name__ == "__main__":
    main()
