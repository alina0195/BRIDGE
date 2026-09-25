#!/usr/bin/env python3
"""Derive ablation targets from an activation-patching results.json.

Rules, applied to the top-k layers of phase 1 using the phase-2 split:
  - attention-dominant layers (attn > mlp, attn >= --min-recovery), strongest
    first: top-3 heads of the first, top-2 of each further layer, stopping at
    --max-heads or at the first head with non-positive recovery
  - MLP-dominant layers (mlp >= attn, mlp >= --min-recovery), strongest
    first: whole MLP, up to --max-mlps

Prints the flags for targeted_ablation.py and probe_survival.py:

    python scripts/mechanistic/extract_ablation_targets.py \
        outputs/mechanistic/mistral/patching/results.json
    # --heads 15:1,15:3,15:7,16:23,16:6 --mlps 19,23,28
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract(path: Path, max_heads: int, max_mlps: int, min_recovery: float):
    data = json.loads(path.read_text())
    top = data["top_layers"]
    attn = {int(k): v for k, v in data["phase2"]["attn_recovery"].items()}
    mlp = {int(k): v for k, v in data["phase2"]["mlp_recovery"].items()}
    head_rec = {int(k): v for k, v in data["phase3"]["head_recovery"].items()}

    attn_layers = sorted(
        [l for l in top if attn.get(l, 0) > mlp.get(l, 0) and attn.get(l, 0) >= min_recovery],
        key=lambda l: -attn[l])
    mlp_layers = sorted(
        [l for l in top if mlp.get(l, 0) >= attn.get(l, 0) and mlp.get(l, 0) >= min_recovery],
        key=lambda l: -mlp[l])

    heads: list[tuple[int, int]] = []
    for i, layer in enumerate(attn_layers):
        if len(heads) >= max_heads:
            break
        recs = head_rec.get(layer, [])
        ranked = sorted(range(len(recs)), key=lambda h: -recs[h])
        for h in ranked[:3 if i == 0 else 2]:
            if recs[h] <= 0:
                break
            heads.append((layer, h))
            if len(heads) >= max_heads:
                break
    return heads, sorted(mlp_layers[:max_mlps])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", type=Path)
    ap.add_argument("--max-heads", type=int, default=5)
    ap.add_argument("--max-mlps", type=int, default=3)
    ap.add_argument("--min-recovery", type=float, default=0.05)
    args = ap.parse_args()
    if not args.results_json.exists():
        sys.exit(f"ERROR: {args.results_json} not found")

    heads, mlps = extract(args.results_json, args.max_heads, args.max_mlps, args.min_recovery)
    parts = []
    if heads:
        parts += ["--heads", ",".join(f"{l}:{h}" for l, h in heads)]
    if mlps:
        parts += ["--mlps", ",".join(str(l) for l in mlps)]
    print(" ".join(parts))


if __name__ == "__main__":
    main()
