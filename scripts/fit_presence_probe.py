#!/usr/bin/env python3
"""Fit the context-presence probe c_hat on the fine-tuned model.

For every volatile eval question two prompts are built, one without context
(label 0) and one with its golden context (label 1).  A logistic probe on the
last prompt token separates them; its unit weight vector is c_hat.

    python scripts/fit_presence_probe.py --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter --layers 31 \
        --output-dir outputs/directions/llama/presence

Writes layer_<L>_presence.npy per layer and probe_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.metrics import load_jsonl  # noqa: E402
from bridge.model import load_model  # noqa: E402
from bridge.probes import ResidualStreamExtractor, fit_logistic_probe  # noqa: E402


def build_prompt(tokenizer, question: str, context: str | None) -> str:
    user = f"Context: {context}\n\n{question}" if context else question
    return tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--layers", default="31", help="Comma-separated layer indices.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.eval_dir / "eval_evolved.jsonl")
    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)
    extractor = ResidualStreamExtractor(model, tokenizer,
                                        [int(x) for x in args.layers.split(",")])

    acts = {l: [] for l in extractor.target_layers}
    labels = []
    for i, rec in enumerate(records):
        for label, ctx in ((0, None), (1, rec["golden_context"])):
            per_layer = extractor.extract_at_positions(
                build_prompt(tokenizer, rec["question"], ctx))
            for l, t in per_layer.items():
                acts[l].append(t.squeeze(0).float().numpy())
            labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(records)}]")
    labels = np.array(labels)

    summary = []
    for l in sorted(acts):
        probe = fit_logistic_probe(np.stack(acts[l]), labels)
        np.save(args.output_dir / f"layer_{l}_presence.npy", probe["direction"])
        summary.append({"layer": l, "cv_accuracy": probe["cv_accuracy"],
                        "cv_f1_macro": probe["cv_f1_macro"], "n": int(labels.size)})
        print(f"  layer {l}: balanced accuracy {probe['cv_accuracy']:.3f}")
    (args.output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
