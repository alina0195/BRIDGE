#!/usr/bin/env python3
"""Targeted-ablation block of the mechanistic summary table.

For each backbone and question class (evolved, stable), from the saved
ablation runs:

    targeted  = VRR_ablated - VRR_baseline  (targeted components)
    random    = mean +- sd of the same quantity over the random-control seeds
    delta_t   = targeted - mean(random)
    |z|       = |delta_t| / sd(random)

The eval has 150 questions per class, so VRR moves in steps of 1/150; a
random-control sd at that floor is flagged, since |z| is then bounded by the
grid rather than by the data.

    python scripts/mechanistic/build_ablation_table.py
Reads  outputs/mechanistic/{bb}/ablation/{targets,random_seed0..2}/results.json
Writes outputs/tables/mech_ablation.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

EVAL_N = 150
SD_FLOOR = 1.0 / EVAL_N
BACKBONES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral3-7B"), ("qwen", "Qwen2.5-7B")]
CLASSES = [("evolved", "evolved_vrr"), ("stable", "stable_vrr")]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _delta(d: dict, key: str) -> float:
    return d["ablated"][key] - d["baseline"][key]


def build(root: Path, seeds: list[int]) -> dict:
    out = {
        "convention": "delta VRR = VRR_ablated - VRR_baseline",
        "sd_floor": SD_FLOOR,
        "eval_n_per_class": EVAL_N,
        "backbones": {},
    }
    for bb, label in BACKBONES:
        abl = root / bb / "ablation"
        td = _load(abl / "targets" / "results.json")
        rds = [_load(abl / f"random_seed{s}" / "results.json") for s in seeds]
        entry = {"targeted_run": str(abl / "targets"),
                 "random_runs": [str(abl / f"random_seed{s}") for s in seeds],
                 "classes": {}}
        for cls, key in CLASSES:
            targeted = _delta(td, key)
            randoms = [_delta(r, key) for r in rds]
            mu = stats.mean(randoms)
            sd = stats.stdev(randoms)
            dt = targeted - mu
            entry["classes"][cls] = {
                "baseline_vrr": td["baseline"][key],
                "ablated_vrr": td["ablated"][key],
                "targeted_delta": targeted,
                "random_deltas": randoms,
                "random_mean": mu,
                "random_sd": sd,
                "delta_tilde": dt,
                "abs_z": abs(dt) / sd if sd else float("inf"),
                "sd_at_resolution_floor": sd <= SD_FLOOR + 1e-9,
            }
        out["backbones"][label] = entry
    return out


def _mag(x: float) -> str:
    """Unsigned 3 dp without the leading zero."""
    return f"{abs(x):.3f}".lstrip("0")


def _fmt(x: float) -> str:
    return ("-" if x < 0 else "+") + _mag(x)


def as_markdown(summary: dict) -> str:
    lines = [
        "# Targeted ablation: evolved and stable",
        "",
        f"Convention: {summary['convention']}. Random control: n=3 seeds, sd with ddof=1. "
        f"VRR resolution floor: 1/{summary['eval_n_per_class']} = {summary['sd_floor']:.4f}.",
        "",
        "| Backbone | Class | Baseline | Ablated | Targeted delta | Random mean +- sd | delta_t | abs(z) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, entry in summary["backbones"].items():
        for cls, c in entry["classes"].items():
            flag = " (*)" if c["sd_at_resolution_floor"] else ""
            lines.append(
                f"| {label} | {cls} | {c['baseline_vrr']:.3f} | {c['ablated_vrr']:.3f} | "
                f"{_fmt(c['targeted_delta'])} | {_fmt(c['random_mean'])} +- {_mag(c['random_sd'])} | "
                f"{_fmt(c['delta_tilde'])} | {c['abs_z']:.1f}{flag} |")
    lines += [
        "",
        "(*) random-control sd is at the resolution floor; |z| is bounded by the "
        "measurement grid, not a calibrated effect size.",
        "",
        "## LaTeX cells (evolved block, then stable block)",
        "",
    ]
    for label, entry in summary["backbones"].items():
        cells = []
        for cls, _ in CLASSES:
            c = entry["classes"][cls]
            if c["abs_z"] < 1:
                zs = "$<1$ n.s."
            else:
                zs = f"${c['abs_z']:.1f}$" + ("**" if c["abs_z"] >= 2.5 else "")
                if c["sd_at_resolution_floor"]:
                    zs += "$^{\\dagger}$"
            cells.append(f"${_fmt(c['targeted_delta'])}$ & ${_fmt(c['random_mean'])} \\pm "
                         f"{_mag(c['random_sd'])}$ & ${_fmt(c['delta_tilde'])}$ & {zs}")
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech-root", type=Path, default=Path("outputs/mechanistic"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/tables"))
    args = ap.parse_args()

    summary = build(args.mech_root, [int(s) for s in args.seeds.split(",")])
    md = as_markdown(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mech_ablation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "mech_ablation.md").write_text(md)
    print(md)
    print(f"wrote {args.out_dir}/mech_ablation.json and .md")


if __name__ == "__main__":
    main()
