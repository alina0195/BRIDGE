#!/usr/bin/env python3
"""Targeted ablation of the components located by activation patching.

The o_proj input slice of each target head and the output of each target MLP
are replaced, at the last prompt token of the prefill pass, by their mean
over --n-mean-samples stable questions (or by zeros).  VRR is measured on
evolved and stable questions before and after, with greedy generation under
the chat template.  Deltas are ablated - baseline.

--random-control replaces the targets by the same number of random heads and
MLPs from layers [--random-layer-min, --random-layer-max], disjoint from the
targets, as a null control (the paper uses seeds 0, 1, 2).

    python scripts/mechanistic/targeted_ablation.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter-path outputs/models/mistral_ca_avmd/adapter \
        --heads 15:1,15:3,15:7,16:23,16:6 --mlps 19,23,28 \
        --limit 150 --output-dir outputs/mechanistic/mistral/ablation/targets

Writes results.json and the four prediction files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge.metrics import (DEFERRAL_PATTERN, TRAINED_DEFERRAL_PATTERN,  # noqa: E402
                            load_jsonl)
from bridge.model import generate, load_model  # noqa: E402
from bridge.prompts import BARE_QA  # noqa: E402
from components import (AblationHookSet, collect_component_means,  # noqa: E402
                        head_dim_of, parse_heads, parse_mlps)


def random_targets(heads, mlps, seed: int, layer_min: int, layer_max: int):
    """Same number of heads and MLPs as the targets, drawn disjoint from them.

    Head indices are drawn from [0, 32) on every backbone; on Qwen (28 heads)
    an index >= 28 selects an empty slice and ablates nothing.
    """
    rng = random.Random(seed)
    pool = list(range(layer_min, layer_max + 1))
    rand_heads: list[tuple[int, int]] = []
    attempts = 0
    while len(rand_heads) < len(heads) and attempts < 10000:
        attempts += 1
        cand = (rng.choice(pool), rng.randrange(0, 32))
        if cand in heads or cand in rand_heads:
            continue
        rand_heads.append(cand)
    mlp_pool = [l for l in pool if l not in set(mlps)]
    rng.shuffle(mlp_pool)
    return sorted(rand_heads), sorted(mlp_pool[:len(mlps)])


def run_generation(records, tokenizer, model, max_new_tokens: int, tag: str):
    out = []
    t0 = time.time()
    for i, rec in enumerate(records):
        prompt = BARE_QA.format(question=rec["question"])
        out.append({"predicted": generate(prompt, tokenizer, model, max_new_tokens),
                    "question": rec["question"]})
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  [{tag}] {i + 1}/{len(records)} ({time.time() - t0:.0f}s)")
    return out


def vrr(preds, pattern) -> float:
    return sum(1 for p in preds if pattern.search(p["predicted"])) / len(preds) if preds else 0.0


def rates(evolved_preds, stable_preds) -> dict:
    """'*_vrr' uses the trained deferral string, '*_vrr_broad' the paper's detector."""
    return {
        "evolved_vrr": vrr(evolved_preds, TRAINED_DEFERRAL_PATTERN),
        "evolved_vrr_broad": vrr(evolved_preds, DEFERRAL_PATTERN),
        "stable_vrr": vrr(stable_preds, TRAINED_DEFERRAL_PATTERN),
        "stable_vrr_broad": vrr(stable_preds, DEFERRAL_PATTERN),
    }


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


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
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ablation-mode", choices=("mean", "zero"), default="mean")
    ap.add_argument("--n-mean-samples", type=int, default=100)
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--random-layer-min", type=int, default=8)
    ap.add_argument("--random-layer-max", type=int, default=28)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heads = sorted(set(parse_heads(args.heads)))
    mlps = sorted(set(parse_mlps(args.mlps)))
    if not heads and not mlps:
        sys.exit("No targets given; use --heads and/or --mlps.")
    if args.random_control:
        heads, mlps = random_targets(heads, mlps, args.random_seed,
                                     args.random_layer_min, args.random_layer_max)
        print(f"[random-control] seed={args.random_seed}")
    print(f"[targets] heads={heads} mlps={mlps} mode={args.ablation_mode}")

    stable = load_jsonl(args.eval_dir / "eval_stable.jsonl")
    evolved = load_jsonl(args.eval_dir / "eval_evolved.jsonl")
    if args.limit:
        stable, evolved = stable[:args.limit], evolved[:args.limit]
    print(f"[data] {len(stable)} stable | {len(evolved)} evolved")

    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)
    head_dim = head_dim_of(model)

    print("\n[baseline]")
    base_ev = run_generation(evolved, tokenizer, model, args.max_new_tokens, "base/evolved")
    base_st = run_generation(stable, tokenizer, model, args.max_new_tokens, "base/stable")
    baseline = rates(base_ev, base_st)

    head_means, mlp_means = {}, {}
    if args.ablation_mode == "mean":
        print(f"\n[means] n={args.n_mean_samples}")
        head_means, mlp_means = collect_component_means(
            model, tokenizer, stable, heads, mlps,
            n_samples=min(args.n_mean_samples, len(stable)), head_dim=head_dim)

    print(f"\n[ablated] {args.ablation_mode}")
    ablation = AblationHookSet(head_means, mlp_means, heads, mlps, head_dim,
                               mode=args.ablation_mode)
    ablation.register(model)
    try:
        abl_ev = run_generation(evolved, tokenizer, model, args.max_new_tokens, "abl/evolved")
        abl_st = run_generation(stable, tokenizer, model, args.max_new_tokens, "abl/stable")
    finally:
        ablation.remove()
    ablated = rates(abl_ev, abl_st)
    delta = {k: ablated[k] - baseline[k] for k in baseline}

    for k in ("evolved_vrr", "stable_vrr"):
        print(f"  {k}: baseline={baseline[k]:.3f} ablated={ablated[k]:.3f} "
              f"delta={delta[k]:+.3f}")

    results = {
        "config": {
            "model": args.model,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "heads": [list(h) for h in heads],
            "mlps": mlps,
            "ablation_mode": args.ablation_mode,
            "random_control": bool(args.random_control),
            "random_seed": args.random_seed if args.random_control else None,
            "limit": args.limit,
            "n_mean_samples": args.n_mean_samples if args.ablation_mode == "mean" else None,
            "max_new_tokens": args.max_new_tokens,
        },
        "baseline": baseline,
        "ablated": ablated,
        "delta": delta,
        "n_interventions": ablation.n_interventions,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_jsonl(args.output_dir / "predictions_evolved_baseline.jsonl", base_ev)
    write_jsonl(args.output_dir / "predictions_evolved_ablated.jsonl", abl_ev)
    write_jsonl(args.output_dir / "predictions_stable_baseline.jsonl", base_st)
    write_jsonl(args.output_dir / "predictions_stable_ablated.jsonl", abl_st)
    print(f"\n[done] {args.output_dir}")


if __name__ == "__main__":
    main()
