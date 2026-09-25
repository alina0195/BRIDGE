#!/usr/bin/env python3
"""Targeted-ablation figure: VRR before and after ablation, evolved and
stable questions, one group per backbone.

    python scripts/plots/plot_targeted_ablation.py
Reads  outputs/mechanistic/{bb}/ablation/targets/results.json
Writes outputs/figures/targeted_ablation_story.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [("qwen", "Qwen-2.5-7B"), ("mistral", "Mistral-7B"), ("llama", "Llama-3.1-8B")]
BASELINE_COLOR = "#B8BCC2"
ABLATED_COLOR = "#2F6FB5"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


def plot(mech_root: Path, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bar_w, gap = 0.18, 0.02
    centers = np.arange(len(MODELS)) * 0.95
    legend = ["Evolved (baseline)", "Evolved (ablated)", "Stable (baseline)", "Stable (ablated)"]
    colors = [BASELINE_COLOR, ABLATED_COLOR, BASELINE_COLOR, ABLATED_COLOR]
    hatches = ["", "", "//", "//"]
    edges = ["#6B7280", "#1E3A5F", "#6B7280", "#1E3A5F"]

    for mi, (bb, _) in enumerate(MODELS):
        d = json.loads((mech_root / bb / "ablation" / "targets" / "results.json").read_text())
        e_base, e_abl = d["baseline"]["evolved_vrr"], d["ablated"]["evolved_vrr"]
        s_base, s_abl = d["baseline"]["stable_vrr"], d["ablated"]["stable_vrr"]
        c = centers[mi]
        pos = [c + k * (bar_w + gap) for k in (-1.5, -0.5, 0.5, 1.5)]
        for j, v in enumerate([e_base, e_abl, s_base, s_abl]):
            ax.bar(pos[j], v, width=bar_w, color=colors[j], edgecolor=edges[j],
                   linewidth=0.8, hatch=hatches[j], label=legend[j] if mi == 0 else None)
        delta = e_abl - e_base
        sign = "+" if delta >= 0 else "-"
        ax.text((pos[0] + pos[1]) / 2, max(e_base, e_abl) + 0.04,
                r"$\mathbf{\Delta{=}{" + sign + "}" + f"{abs(delta):.2f}" + "}$",
                ha="center", va="bottom", fontsize=11.5, color=ABLATED_COLOR)

    ax.set_xticks(centers)
    ax.set_xticklabels([lbl for _, lbl in MODELS])
    ax.set_ylabel("VRR")
    ax.set_ylim(0.0, 0.9)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech-root", type=Path, default=Path("outputs/mechanistic"))
    ap.add_argument("--out", type=Path, default=Path("outputs/figures/targeted_ablation_story"),
                    help="Output path without extension.")
    args = ap.parse_args()
    plot(args.mech_root, args.out)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
