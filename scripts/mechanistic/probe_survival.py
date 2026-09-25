#!/usr/bin/env python3
"""Probe survival: does the volatility signal survive targeted ablation?

Residuals at the last token of the bare prompt are captured at every layer,
without and with the targeted ablation of targeted_ablation.py, and a
balanced logistic probe (stable vs evolved, 5-fold CV) is fitted per layer
on each.  Accuracy that is preserved downstream of the ablated components
indicates a distributed representation rather than a narrow circuit.

    python scripts/mechanistic/probe_survival.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter-path outputs/models/mistral_ca_avmd/adapter \
        --heads 15:1,15:3,15:7,16:23,16:6 --mlps 19,23,28 \
        --limit 200 --output-dir outputs/mechanistic/mistral/probe_survival

Writes results.json (baseline_probes, ablated_probes, peaks).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge.metrics import load_jsonl  # noqa: E402
from bridge.model import get_layers, input_device, load_model  # noqa: E402
from components import (AblationHookSet, bare_input_ids,  # noqa: E402
                        collect_component_means, head_dim_of, parse_heads, parse_mlps)


@torch.inference_mode()
def capture_residuals(model, tokenizer, records, tag: str) -> np.ndarray:
    """Last-token block outputs at every layer, shape (n_records, n_layers, hidden)."""
    device = input_device(model)
    layers = get_layers(model)
    buf: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                buf[i] = h[0, -1, :].detach().float().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    captured = []
    try:
        t0 = time.time()
        for j, rec in enumerate(records):
            buf.clear()
            model(input_ids=bare_input_ids(tokenizer, rec["question"], device))
            captured.append(np.stack([buf[i].numpy() for i in range(len(layers))]))
            if (j + 1) % 50 == 0 or (j + 1) == len(records):
                print(f"  [{tag}] {j + 1}/{len(records)} ({time.time() - t0:.0f}s)")
    finally:
        for h in handles:
            h.remove()
    return np.stack(captured)


def fit_probes_per_layer(acts_stable: np.ndarray, acts_evolved: np.ndarray) -> list[dict]:
    """Mean and std of 5-fold balanced accuracy per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate

    labels = np.concatenate([np.zeros(len(acts_stable)), np.ones(len(acts_evolved))]).astype(int)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = []
    for layer in range(acts_stable.shape[1]):
        X = np.concatenate([acts_stable[:, layer, :], acts_evolved[:, layer, :]], axis=0)
        scores = cross_validate(clf, X, labels, cv=cv, scoring="balanced_accuracy", n_jobs=1)
        acc = scores["test_score"]
        out.append({"layer": layer, "cv_accuracy": float(acc.mean()),
                    "cv_std": float(acc.std())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--heads", default="", help='Target heads, "L:H,L:H".')
    ap.add_argument("--mlps", default="", help='Target MLP layers, "L,L".')
    ap.add_argument("--limit", type=int, default=None, help="Questions per class.")
    ap.add_argument("--ablation-mode", choices=("mean", "zero"), default="mean")
    ap.add_argument("--n-mean-samples", type=int, default=100)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heads = sorted(set(parse_heads(args.heads)))
    mlps = sorted(set(parse_mlps(args.mlps)))
    if not heads and not mlps:
        sys.exit("No targets given; use --heads and/or --mlps.")
    print(f"[targets] heads={heads} mlps={mlps} mode={args.ablation_mode}")

    stable = load_jsonl(args.eval_dir / "eval_stable.jsonl")
    evolved = load_jsonl(args.eval_dir / "eval_evolved.jsonl")
    if args.limit:
        stable, evolved = stable[:args.limit], evolved[:args.limit]
    print(f"[data] {len(stable)} stable | {len(evolved)} evolved")

    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)
    head_dim = head_dim_of(model)
    n_layers = len(get_layers(model))

    print("\n[baseline capture]")
    base_st = capture_residuals(model, tokenizer, stable, "base/stable")
    base_ev = capture_residuals(model, tokenizer, evolved, "base/evolved")

    head_means, mlp_means = {}, {}
    if args.ablation_mode == "mean":
        print(f"\n[means] n={args.n_mean_samples}")
        head_means, mlp_means = collect_component_means(
            model, tokenizer, stable, heads, mlps,
            n_samples=min(args.n_mean_samples, len(stable)), head_dim=head_dim)

    print(f"\n[ablated capture] {args.ablation_mode}")
    ablation = AblationHookSet(head_means, mlp_means, heads, mlps, head_dim,
                               mode=args.ablation_mode)
    ablation.register(model)
    try:
        abl_st = capture_residuals(model, tokenizer, stable, "abl/stable")
        abl_ev = capture_residuals(model, tokenizer, evolved, "abl/evolved")
    finally:
        ablation.remove()

    print(f"\n[probe] logistic probes at {n_layers} layers")
    baseline = fit_probes_per_layer(base_st, base_ev)
    ablated = fit_probes_per_layer(abl_st, abl_ev)
    for b, a in zip(baseline, ablated):
        print(f"  L{b['layer']:>2}  baseline={b['cv_accuracy']:.3f}  "
              f"ablated={a['cv_accuracy']:.3f}  delta={a['cv_accuracy'] - b['cv_accuracy']:+.3f}")

    results = {
        "config": {
            "model": args.model,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "heads": [list(h) for h in heads],
            "mlps": mlps,
            "ablation_mode": args.ablation_mode,
            "limit": args.limit,
            "n_layers": n_layers,
            "hidden_size": int(model.config.hidden_size),
        },
        "baseline_probes": baseline,
        "ablated_probes": ablated,
        "peak_baseline": max(baseline, key=lambda r: r["cv_accuracy"]),
        "peak_ablated": max(ablated, key=lambda r: r["cv_accuracy"]),
        "n_interventions": ablation.n_interventions,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
