#!/usr/bin/env python3
"""Activation patching (causal tracing) of the volatile/stable distinction.

Source run: an evolved question (the model defers).  Baseline run: a stable
question matched by entity_key (the model answers).  One activation of the
source run, at the last prompt token, is written into the baseline run and
the probability of the first deferral token is read at the next position:

    path_recovery = (P_patch - P_base) / (P_src - P_base)

Pairs with P_src - P_base < --min-gap are skipped.

    phase 1   residual stream, every layer
    phase 2   attention output vs MLP output, top-k layers of phase 1
    phase 3   each attention head (slice of the o_proj input), top-k layers

    python scripts/mechanistic/activation_patching.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter \
        --output-dir outputs/mechanistic/llama/patching \
        --n-pairs 50 --top-k-layers 6 --min-gap 0.02

Writes results.json (config, phase1, top_layers, phase2, phase3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge.metrics import load_jsonl  # noqa: E402
from bridge.model import get_layers, input_device, load_model  # noqa: E402
from components import bare_input_ids, get_attn, get_mlp, get_o_proj  # noqa: E402

# P(defer) is the probability of the first token of the trained deferral
# string; the leading space matches generation right after "Answer:".
DEFERRAL_PHRASE_PREFIX = " I don't have"


@dataclass
class LayerCache:
    residual: torch.Tensor     # block output
    attn_out: torch.Tensor     # self-attention output (before the residual add)
    attn_pre_o: torch.Tensor   # o_proj input: concatenated head outputs
    mlp_out: torch.Tensor      # MLP output (before the residual add)


@dataclass
class ActivationCache:
    layers: dict[int, LayerCache] = field(default_factory=dict)
    p_defer: float = 0.0


def deferral_token_id(tokenizer, prefix: str = DEFERRAL_PHRASE_PREFIX) -> int:
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    tok = tokenizer.convert_ids_to_tokens([ids[0]])[0]
    print(f"[deferral] prefix={prefix!r} first token={tok!r} id={ids[0]}")
    return ids[0]


@torch.inference_mode()
def cache_activations(model, tokenizer, question: str, defer_id: int,
                      layers: list[int]) -> ActivationCache:
    """One forward pass; cache all four components at the last prompt token."""
    input_ids = bare_input_ids(tokenizer, question, input_device(model))
    last = input_ids.shape[1] - 1
    raw: dict[int, dict[str, torch.Tensor]] = {i: {} for i in layers}
    handles = []
    blocks = get_layers(model)

    def out_hook(idx, key):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            raw[idx][key] = h[0, last, :].detach().cpu()
        return hook

    def o_pre_hook(idx):
        def hook(module, inp):
            raw[idx]["attn_pre_o"] = inp[0][0, last, :].detach().cpu()
        return hook

    for idx in layers:
        block = blocks[idx]
        attn = get_attn(block)
        handles.append(block.register_forward_hook(out_hook(idx, "residual")))
        handles.append(attn.register_forward_hook(out_hook(idx, "attn_out")))
        handles.append(get_o_proj(attn).register_forward_pre_hook(o_pre_hook(idx)))
        handles.append(get_mlp(block).register_forward_hook(out_hook(idx, "mlp_out")))
    try:
        logits = model(input_ids=input_ids).logits[0, last, :]
        p_defer = float(torch.softmax(logits, dim=-1)[defer_id].cpu())
    finally:
        for h in handles:
            h.remove()

    cache = ActivationCache(p_defer=p_defer)
    for idx in layers:
        cache.layers[idx] = LayerCache(**{k: raw[idx][k] for k in
                                          ("residual", "attn_out", "attn_pre_o", "mlp_out")})
    return cache


@torch.inference_mode()
def patch_and_measure(model, tokenizer, base_question: str, src: ActivationCache,
                      layer: int, component: str, defer_id: int, n_heads: int) -> float:
    """P(defer) on the baseline prompt with one source activation patched in.

    component: "residual" | "attn_out" | "mlp_out" | "head_<N>"
    """
    device = input_device(model)
    input_ids = bare_input_ids(tokenizer, base_question, device)
    last = input_ids.shape[1] - 1
    block = get_layers(model)[layer]
    attn = get_attn(block)
    src_layer = src.layers[layer]

    def replace_output(value):
        value = value.to(device)

        def hook(module, inp, out):
            h = out[0].clone() if isinstance(out, tuple) else out.clone()
            h[0, last, :] = value
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    if component == "residual":
        handle = block.register_forward_hook(replace_output(src_layer.residual))
    elif component == "attn_out":
        handle = attn.register_forward_hook(replace_output(src_layer.attn_out))
    elif component == "mlp_out":
        handle = get_mlp(block).register_forward_hook(replace_output(src_layer.mlp_out))
    elif component.startswith("head_"):
        head = int(component.split("_")[1])
        d = src_layer.attn_pre_o.shape[0] // n_heads
        src_head = src_layer.attn_pre_o[head * d:(head + 1) * d].to(device)

        def pre_hook(module, inp):
            x = inp[0].clone()
            x[0, last, head * d:(head + 1) * d] = src_head
            return (x,) + inp[1:]
        handle = get_o_proj(attn).register_forward_pre_hook(pre_hook)
    else:
        raise ValueError(f"Unknown component: {component!r}")

    try:
        logits = model(input_ids=input_ids).logits[0, last, :]
        return float(torch.softmax(logits, dim=-1)[defer_id].cpu())
    finally:
        handle.remove()


def load_pairs(eval_dir: Path, n_pairs: int | None) -> list[tuple[dict, dict]]:
    """(evolved, stable) pairs matched by entity_key; unmatched evolved
    questions are paired with shuffled stable questions (seed 42)."""
    evolved = load_jsonl(eval_dir / "eval_evolved.jsonl")
    stable = load_jsonl(eval_dir / "eval_stable.jsonl")
    by_entity: dict[str, list[dict]] = {}
    for rec in stable:
        by_entity.setdefault(rec.get("entity_key", ""), []).append(rec)

    pairs, unmatched = [], []
    for rec in evolved:
        k = rec.get("entity_key", "")
        if k and k in by_entity:
            pairs.append((rec, by_entity[k][0]))
        else:
            unmatched.append(rec)

    pool = [r for recs in by_entity.values() for r in recs]
    np.random.seed(42)
    np.random.shuffle(pool)
    for i, rec in enumerate(unmatched):
        pairs.append((rec, pool[i % len(pool)]))
    print(f"[data] {len(evolved)} evolved, {len(stable)} stable -> {len(pairs)} pairs "
          f"({len(pairs) - len(unmatched)} matched by entity_key)")
    return pairs[:n_pairs] if n_pairs else pairs


def run_phase(model, tokenizer, pairs, defer_id, layers, components, n_heads,
              min_gap, tag):
    """Mean path recovery per (layer, component) over pairs with a valid gap."""
    rec = {(l, c): [] for l in layers for c in components}
    n_valid = 0
    t0 = time.time()
    for i, (ev, st) in enumerate(pairs):
        src = cache_activations(model, tokenizer, ev["question"], defer_id, layers)
        base = cache_activations(model, tokenizer, st["question"], defer_id, layers)
        gap = src.p_defer - base.p_defer
        if gap < min_gap:
            continue
        n_valid += 1
        for l in layers:
            for c in components:
                p = patch_and_measure(model, tokenizer, st["question"], src, l, c,
                                      defer_id, n_heads)
                rec[(l, c)].append((p - base.p_defer) / gap)
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i + 1}/{len(pairs)} pairs ({time.time() - t0:.0f}s)")
    return rec, n_valid


def mean_or_zero(v):
    return float(np.mean(v)) if v else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--top-k-layers", type=int, default=6,
                    help="Layers carried from phase 1 into phases 2 and 3.")
    ap.add_argument("--min-gap", type=float, default=0.02,
                    help="Minimum P_src - P_base for a pair to count.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)
    n_layers = len(get_layers(model))
    n_heads = int(model.config.num_attention_heads)
    defer_id = deferral_token_id(tokenizer)
    pairs = load_pairs(args.eval_dir, args.n_pairs)

    results = {"config": {
        "model": args.model,
        "adapter_path": str(args.adapter_path),
        "n_pairs": len(pairs),
        "top_k_layers": args.top_k_layers,
        "min_gap": args.min_gap,
        "deferral_token_id": defer_id,
        "deferral_phrase_prefix": DEFERRAL_PHRASE_PREFIX,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }}

    print("\n[phase 1] residual stream, all layers")
    all_layers = list(range(n_layers))
    rec, n_valid = run_phase(model, tokenizer, pairs, defer_id, all_layers,
                             ["residual"], n_heads, args.min_gap, "p1")
    results["phase1"] = {
        "layers": all_layers,
        "recovery_mean": [mean_or_zero(rec[(l, "residual")]) for l in all_layers],
        "recovery_std": [float(np.std(rec[(l, "residual")])) if rec[(l, "residual")] else 0.0
                         for l in all_layers],
        "n_valid_pairs": n_valid,
        "n_total_pairs": len(pairs),
    }
    ranked = sorted(zip(all_layers, results["phase1"]["recovery_mean"]), key=lambda x: -x[1])
    top = sorted(l for l, _ in ranked[:args.top_k_layers])
    results["top_layers"] = top
    print(f"[phase 1] {n_valid}/{len(pairs)} valid pairs; top layers {top}")

    print(f"\n[phase 2] attention vs MLP at layers {top}")
    rec, n_valid = run_phase(model, tokenizer, pairs, defer_id, top,
                             ["attn_out", "mlp_out"], n_heads, args.min_gap, "p2")
    results["phase2"] = {
        "top_layers": top,
        "attn_recovery": {str(l): mean_or_zero(rec[(l, "attn_out")]) for l in top},
        "mlp_recovery": {str(l): mean_or_zero(rec[(l, "mlp_out")]) for l in top},
        "n_valid_pairs": n_valid,
    }
    for l in top:
        print(f"  L{l}: attn={results['phase2']['attn_recovery'][str(l)]:.3f} "
              f"mlp={results['phase2']['mlp_recovery'][str(l)]:.3f}")

    print(f"\n[phase 3] {n_heads} heads x {len(top)} layers")
    heads = [f"head_{h}" for h in range(n_heads)]
    rec, n_valid = run_phase(model, tokenizer, pairs, defer_id, top, heads,
                             n_heads, args.min_gap, "p3")
    results["phase3"] = {
        "top_layers": top,
        "n_heads": n_heads,
        "head_recovery": {str(l): [mean_or_zero(rec[(l, h)]) for h in heads] for l in top},
        "n_valid_pairs": n_valid,
    }

    out = args.output_dir / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
