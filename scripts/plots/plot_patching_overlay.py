#!/usr/bin/env python3
"""Activation-patching overlay: residual-stream path recovery against
relative depth (layer / (n_layers - 1)) for all backbones, +-1 std band,
with the peak layer of each backbone annotated.

    python scripts/plots/plot_patching_overlay.py
Reads  outputs/mechanistic/{bb}/patching/results.json
Writes outputs/figures/activation_patching_overlay.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import BACKBONE_COLORS, BACKBONE_LABELS  # noqa: E402

BACKBONES = ("llama", "mistral", "qwen")
# Broad mid-layer windows of elevated recovery, shaded per backbone.
PEAK_BANDS = {"llama": (15, 29), "mistral": (15, 28), "qwen": (19, 23)}
# Offsets (in data units) of the peak-layer labels.
ANNOT_OFFSETS = {"llama": (-0.09, 0.30), "mistral": (0.03, 0.45), "qwen": (0.05, -0.18)}

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def plot(mech_root: Path, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for bb in BACKBONES:
        d = json.loads((mech_root / bb / "patching" / "results.json").read_text())
        layers = np.array(d["phase1"]["layers"], dtype=float)
        rec = np.array(d["phase1"]["recovery_mean"])
        std = np.array(d["phase1"]["recovery_std"])
        n_layers = d["config"]["n_layers"]
        depth = layers / (n_layers - 1)
        c = BACKBONE_COLORS[bb]

        ax.fill_between(depth, rec - std, rec + std, color=c, alpha=0.12, linewidth=0)
        ax.plot(depth, rec, color=c, linewidth=2.0, marker="o", markersize=3.5,
                label=BACKBONE_LABELS[bb])
        lo, hi = PEAK_BANDS[bb]
        ax.axvspan(lo / (n_layers - 1), hi / (n_layers - 1), color=c, alpha=0.06, zorder=0)

        i = int(np.argmax(rec))
        dx, dy = ANNOT_OFFSETS[bb]
        ax.annotate(f"L{int(layers[i])}", xy=(depth[i], rec[i]), xycoords="data",
                    xytext=(depth[i] + dx, rec[i] + dy), textcoords="data",
                    fontsize=10, fontweight="bold", color=c, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.9, shrinkA=0, shrinkB=2))

    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.3, 3.6)
    ax.yaxis.grid(True, color="#D1D5DB", linewidth=0.7, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Relative depth (layer index / n_layers)", fontsize=12)
    ax.set_ylabel("Path recovery", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False,
              prop={"size": 13})

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech-root", type=Path, default=Path("outputs/mechanistic"))
    ap.add_argument("--out", type=Path, default=Path("outputs/figures/activation_patching_overlay"),
                    help="Output path without extension.")
    args = ap.parse_args()
    plot(args.mech_root, args.out)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
