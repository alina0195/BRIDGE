#!/usr/bin/env python3
"""Fit v_hat at the hook layer and derive the volatility thresholds tau_v.

v_hat = (mu_volatile - mu_stable) / ||mu_volatile - mu_stable|| is fitted on
the last-prompt-token activations of the hook layer, on the bare evaluation
prompt and the chat template, i.e. exactly what the hook reads at test time.
gap = ||mu_volatile - mu_stable|| is the dose unit (alpha = k * gap).

tau_v is set by quantile: for each rate r in --rates, the threshold at which a
share r of stable prompts passes the volatility condition.

    python scripts/fit_hook_direction.py --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter --layer 31 \
        --output-dir outputs/directions/llama/hook

    # shuffled-label control, from the cached activations (CPU only)
    python scripts/fit_hook_direction.py --layer 31 --shuffle-labels --fit-seed 0 \
        --output-dir outputs/directions/llama/hook

Outputs (<tag> is empty for v_hat, _shuf<seed> for the control):
    layer_<L>_vhat<tag>.npy          unit direction (float32)
    layer_<L>_vhat<tag>.json         gap, tau_v per rate, projection statistics
    layer_<L>_projections<tag>.npz   per-prompt projections and labels
    layer_<L>_acts.npz               activation cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.metrics import load_jsonl  # noqa: E402
from bridge.probes import fit_mean_difference, tau_at_stable_pass  # noqa: E402
from bridge.prompts import BARE_QA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--layer", type=int, required=True, help="Must equal the hook layer.")
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rates", default="0.40,0.30,0.20",
                    help="Stable pass rates at which tau_v is reported.")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="Permute the stable/volatile labels before fitting (control).")
    ap.add_argument("--fit-seed", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = args.output_dir / f"layer_{args.layer}_acts.npz"
    if cache.exists():
        z = np.load(cache)
        acts, labels = z["acts"], z["labels"]
        print(f"[acts] loaded {acts.shape} from {cache}")
    else:
        if not args.base_model:
            ap.error("--base-model is required when no activation cache exists")
        from bridge.model import load_model, resolve_layer
        from bridge.probes import collect_hook_activations

        stable = load_jsonl(args.eval_dir / "eval_stable.jsonl")
        evolved = load_jsonl(args.eval_dir / "eval_evolved.jsonl")
        tokenizer, model = load_model(args.base_model, adapter_path=args.adapter_path)
        layer = resolve_layer(model, args.layer)
        prompts = [BARE_QA.format(question=r["question"]) for r in stable + evolved]
        acts = collect_hook_activations(model, tokenizer, prompts, layer)
        labels = np.array([0] * len(stable) + [1] * len(evolved))
        np.savez(cache, acts=acts.astype(np.float32), labels=labels, layer=layer)

    fit_labels = labels
    tag = ""
    if args.shuffle_labels:
        perm = np.random.default_rng([args.fit_seed, 23]).permutation(labels.size)
        fit_labels, tag = labels[perm], f"_shuf{args.fit_seed}"

    fit = fit_mean_difference(acts, fit_labels)
    unit = fit.pop("unit").astype(np.float32)
    proj = acts.astype(np.float64) @ unit.astype(np.float64)
    stable_proj = proj[labels == 0]
    volatile_proj = proj[labels == 1]
    rates = [float(r) for r in args.rates.split(",")]
    taus = {f"{r:.2f}": tau_at_stable_pass(stable_proj, r) for r in rates}

    np.save(args.output_dir / f"layer_{args.layer}_vhat{tag}.npy", unit)
    np.savez(args.output_dir / f"layer_{args.layer}_projections{tag}.npz",
             projections=proj, labels=labels)
    meta = {
        "layer": args.layer,
        "definition": "v_hat = (mu_volatile - mu_stable) / ||mu_volatile - mu_stable||",
        "shuffled_labels": args.shuffle_labels,
        "fit_seed": args.fit_seed if args.shuffle_labels else None,
        "n_stable": int((labels == 0).sum()),
        "n_volatile": int((labels == 1).sum()),
        **fit,
        "tau_v_at_stable_pass": taus,
        "volatile_pass_at_tau_v": {r: float((volatile_proj > t).mean())
                                   for r, t in taus.items()},
    }
    (args.output_dir / f"layer_{args.layer}_vhat{tag}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"[fit] gap={fit['gap']:.4f} mean|h|={fit['mean_activation_norm']:.2f}")
    for r, t in taus.items():
        print(f"  stable pass {r}: tau_v={t:+.4f}  volatile pass "
              f"{meta['volatile_pass_at_tau_v'][r]:.1%}")


if __name__ == "__main__":
    main()
