#!/usr/bin/env python3
"""Probe-survival figure: volatility-probe accuracy per layer before (solid)
and after (dashed) targeted ablation, all backbones, rolling-mean smoothed.

    python scripts/plots/plot_probe_survival.py
Reads  outputs/mechanistic/{bb}/probe_survival/results.json
Writes outputs/figures/probe_survival_combined.{pdf,png}
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
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import BACKBONE_COLORS, BACKBONE_LABELS, BACKBONE_MARKERS  # noqa: E402

BACKBONES = ("llama", "mistral", "qwen")
LABELS = {**BACKBONE_LABELS, "qwen": "Qwen-2.5-7B"}
SMOOTH_WINDOW = 3   # rolling-mean half-width
MARK_EVERY = 5


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Symmetric rolling mean with edge padding (length preserving)."""
    if w <= 1:
        return y
    yp = np.pad(y, w, mode="edge")
    return np.convolve(yp, np.ones(2 * w + 1) / (2 * w + 1), mode="same")[w:-w]


def plot(mech_root: Path, out_path: Path) -> None:
    runs = []
    for bb in BACKBONES:
        d = json.loads((mech_root / bb / "probe_survival" / "results.json").read_text())
        runs.append({
            "label": LABELS[bb], "color": BACKBONE_COLORS[bb], "marker": BACKBONE_MARKERS[bb],
            "layers": np.array([p["layer"] for p in d["baseline_probes"]]),
            "acc_b": smooth(np.array([p["cv_accuracy"] for p in d["baseline_probes"]]), SMOOTH_WINDOW),
            "acc_a": smooth(np.array([p["cv_accuracy"] for p in d["ablated_probes"]]), SMOOTH_WINDOW),
        })
    l_max = max(int(r["layers"].max()) for r in runs)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for r in runs:
        common = dict(color=r["color"], marker=r["marker"], markersize=4.5,
                      markevery=MARK_EVERY, markeredgecolor="white", markeredgewidth=0.6)
        ax.plot(r["layers"], r["acc_b"], linestyle="-", linewidth=1.8, zorder=3, **common)
        ax.plot(r["layers"], r["acc_a"], linestyle="--", linewidth=1.6, alpha=0.95,
                zorder=2, **common)

    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":", zorder=1)
    ax.text(l_max, 0.5, " chance", fontsize=6.8, color="#666", ha="right", va="bottom")
    ax.set_xlabel("Layer", fontsize=8.5)
    ax.set_ylabel("Linear-probe accuracy", fontsize=8.5)
    ax.set_ylim(0.45, 0.90)
    ax.set_xlim(-0.5, l_max + 0.5)
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.yaxis.grid(True, color="#D6D8DB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    backbone_handles = [
        Line2D([0], [0], color=r["color"], marker=r["marker"], markersize=5,
               markeredgecolor="white", markeredgewidth=0.6, linewidth=1.8, label=r["label"])
        for r in runs]
    style_handles = [
        Line2D([0], [0], color="#444", linestyle="-", linewidth=1.6, label="baseline"),
        Line2D([0], [0], color="#444", linestyle="--", linewidth=1.6, label="ablated")]
    fig.legend(handles=backbone_handles, fontsize=9, loc="lower center",
               ncol=len(backbone_handles), frameon=False, handlelength=2.4,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.8)
    ax.legend(handles=style_handles, fontsize=9, loc="upper right", ncol=1,
              frameon=False, handlelength=2.4, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.subplots_adjust(bottom=0.16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech-root", type=Path, default=Path("outputs/mechanistic"))
    ap.add_argument("--out", type=Path, default=Path("outputs/figures/probe_survival_combined"),
                    help="Output path without extension.")
    args = ap.parse_args()
    plot(args.mech_root, args.out)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
