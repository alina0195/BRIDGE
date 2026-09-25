#!/usr/bin/env python3
"""Evaluate FLARE on the behavioral tests.

Writes test1..test4 .json + .predictions.jsonl in the bridge.evaluation
format, so bridge.metrics.run_metrics(output_dir) gives SR/SDR/VRR/DD/CO.

Retrieval per test (see bridge.baselines.protocol):
    test 1   FLARE loop, no passage: SR/SDR on the bare question, the same
             condition as the test-4 vacuum.  This is the default and the
             protocol of the main-table comparison.
    test 2   FLARE loop, gold passage retrieved when the trigger fires.
    test 3   passage always provided (RAG_QA).
    test 4   vacuum: FLARE loop, no passage (VRR); grounded: passage always
             provided (RAG_QA_GROUNDED).
    --oracle-t1 additionally runs test 1 with the gold passage retrieved on
             trigger (test1_stable_retention_oracle.*); run_metrics ignores it.

Example
-------
    python scripts/baselines/run_flare.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval-dir data/evowiki --output-dir outputs/baselines/llama/flare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.baselines.flare import flare_generate, generate_with_logprobs  # noqa: E402
from bridge.baselines.protocol import run_baseline  # noqa: E402
from bridge.model import load_model  # noqa: E402
from bridge.prompts import RAG_QA, RAG_QA_GROUNDED  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HuggingFace model id of the backbone.")
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tests", default="1,2,3,4")
    ap.add_argument("--oracle-t1", action="store_true",
                    help="Also run test 1 with gold-passage retrieval on trigger.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--filter-threshold", type=float, default=0.5,
                    help="Retrieve if any look-ahead token probability is below this.")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="Maximum sentence-level look-ahead steps.")
    args = ap.parse_args()

    tokenizer, model = load_model(args.model)

    def adaptive(instruction, paragraph):
        return flare_generate(model, tokenizer, instruction, paragraph,
                              filter_threshold=args.filter_threshold,
                              max_iterations=args.max_iterations,
                              max_new_tokens=args.max_new_tokens)

    def grounded(question, context, test):
        template = RAG_QA if test == 3 else RAG_QA_GROUNDED
        text, _, _ = generate_with_logprobs(
            template.format(context=context, question=question), tokenizer, model,
            max_new_tokens=args.max_new_tokens)
        return text.strip(), {}

    run_baseline(adaptive, grounded, args.eval_dir, args.output_dir,
                 {int(t) for t in args.tests.split(",")}, oracle_t1=args.oracle_t1,
                 config={"method": "flare", "model": args.model,
                         "filter_threshold": args.filter_threshold,
                         "max_iterations": args.max_iterations,
                         "max_new_tokens": args.max_new_tokens})


if __name__ == "__main__":
    main()
