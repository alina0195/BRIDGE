#!/usr/bin/env python3
"""Linear volatility probe per layer (Section 3.2, "linear probing").

For every stable and evolved eval question the residual stream is read at two
positions of the bare prompt (no chat template):

    prompt_last    last prompt token ("Answer:"), before any generation
    answer_first   first token of the gold answer under teacher forcing
                   (input = prompt + " " + gold answer)

A balanced logistic probe (5-fold CV) is fitted per layer and position.
Run once per model variant (zero-shot and each fine-tuned adapter); the
saved unit directions feed the cosine-similarity heatmap.

    python scripts/mechanistic/volatility_probe.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter \
        --output-dir outputs/mechanistic/llama/probes/ca_avmd

Outputs (in --output-dir):
    probe_summary.json                       per layer x position accuracy
    config.json
    activations/labels.npy                   0 = stable, 1 = evolved
    activations/layer_<L>_<pos>_acts.npy     (N, hidden) residuals
    activations/layer_<L>_<pos>_direction.npy  unit probe weight vector
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.metrics import load_jsonl  # noqa: E402
from bridge.model import load_model  # noqa: E402
from bridge.probes import ResidualStreamExtractor, fit_logistic_probe  # noqa: E402
from bridge.prompts import BARE_QA  # noqa: E402

EXTRACTION_POINTS = ("prompt_last", "answer_first")


def extract_activations(extractor, tokenizer, stable, evolved, point):
    """Residuals per layer, shape (N, hidden), and labels (0 stable, 1 evolved)."""
    records = [(r, 0) for r in stable] + [(r, 1) for r in evolved]
    labels = np.array([lbl for _, lbl in records])
    acts = {l: [] for l in extractor.target_layers}

    t0 = time.time()
    for i, (rec, _) in enumerate(records):
        prompt = BARE_QA.format(question=rec["question"])
        gold = rec.get("latest_answer") or rec.get("answer", "")
        if point == "answer_first" and gold:
            # Position of the first answer token = length of the tokenised prompt.
            prompt_len = len(tokenizer(prompt, return_tensors="pt",
                                       add_special_tokens=True).input_ids[0])
            per_layer = extractor.extract_at_positions(prompt + " " + gold,
                                                       token_positions=[prompt_len])
        else:
            per_layer = extractor.extract_at_positions(prompt, token_positions=None)
        for layer, tensor in per_layer.items():
            acts[layer].append(tensor.squeeze(0).numpy())
        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            print(f"  [{i + 1}/{len(records)}] {time.time() - t0:.1f}s")

    return {l: np.stack(v) for l, v in acts.items()}, labels


def fit_probes(activations, labels, act_dir: Path, point: str) -> list[dict]:
    rows = []
    for layer in sorted(activations):
        acts = activations[layer]
        np.save(act_dir / f"layer_{layer}_{point}_acts.npy", acts)
        probe = fit_logistic_probe(acts, labels)
        np.save(act_dir / f"layer_{layer}_{point}_direction.npy", probe["direction"])
        print(f"  layer {layer:>3d} | bal_acc={probe['cv_accuracy']:.3f} "
              f"F1={probe['cv_f1_macro']:.3f}")
        rows.append({
            "layer": layer,
            "extraction_point": point,
            "logistic_cv_accuracy": probe["cv_accuracy"],
            "logistic_cv_f1_macro": probe["cv_f1_macro"],
            "logistic_intercept": probe["intercept"],
            "n_samples": int(len(labels)),
            "n_stable": int((labels == 0).sum()),
            "n_evolved": int(labels.sum()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None, help="Omit for zero-shot.")
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-layers", default="-1,-4,-8,-16",
                    help="Comma-separated layer indices (negative = from the end).")
    ap.add_argument("--extraction-points", default=",".join(EXTRACTION_POINTS))
    ap.add_argument("--limit", type=int, default=None, help="Samples per class.")
    args = ap.parse_args()

    act_dir = args.output_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.target_layers.split(",")]
    points = [x.strip() for x in args.extraction_points.split(",")]
    for p in points:
        if p not in EXTRACTION_POINTS:
            raise SystemExit(f"unknown extraction point {p!r}")

    stable = load_jsonl(args.eval_dir / "eval_stable.jsonl")
    evolved = load_jsonl(args.eval_dir / "eval_evolved.jsonl")
    if args.limit:
        stable, evolved = stable[:args.limit], evolved[:args.limit]
    print(f"[data] {len(stable)} stable | {len(evolved)} evolved")

    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)
    extractor = ResidualStreamExtractor(model, tokenizer, target_layers)
    np.save(act_dir / "labels.npy", np.array([0] * len(stable) + [1] * len(evolved)))

    summary = []
    for point in points:
        print(f"\n[extract] {point}")
        activations, labels = extract_activations(extractor, tokenizer, stable, evolved, point)
        summary.extend(fit_probes(activations, labels, act_dir, point))

    (args.output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps({
        "model": args.model,
        "adapter_path": args.adapter_path,
        "eval_dir": str(args.eval_dir),
        "target_layers": target_layers,
        "extraction_points": points,
        "limit": args.limit,
        "n_stable": len(stable),
        "n_evolved": len(evolved),
    }, indent=2))
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
