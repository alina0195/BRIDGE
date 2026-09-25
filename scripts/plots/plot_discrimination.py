#!/usr/bin/env python3
"""Deferral discrimination figures, from outputs/tables/results.json.

    (a) the SDR / VRR plane: colour is the arm, shape is the backbone; the
        dashed diagonal is indiscriminate deferral and the height above it is DD
    (b) change against CA-AVMD for the two gated arms: hatched bars down are
        delta SR, solid bars up are delta DD, on one shared scale

    python scripts/make_tables.py && python scripts/plots/plot_discrimination.py
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

BACKBONES = [("llama", "Llama-3.1-8B", "o"), ("mistral", "Mistral3-7B", "s"),
             ("qwen", "Qwen2.5-7B", "^")]
ARMS = {  # key: (legend label, colour)
    "zero_shot": ("Zero-shot", "#b0aea3"),
    "prompted": ("Prompted to abstain", "#b0aea3"),
    "sft": ("SFT", "#6f6d64"),
    "ca_avmd": ("CA-AVMD", "#90aefe"),
    "presence_only": ("+ presence gate", "#f49194"),
    "bridge": ("+ BRIDGe", "#27b554"),
}
INK, INK_SOFT, RULE, GRID = "#44433D", "#77756C", "#C9C7BD", "#EAE8E2"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def shade(colour: str, k: float = 0.62) -> str:
    """A darker step of the same hue, for mark edges."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(colour))
    return to_hex(colorsys.hls_to_rgb(h, l * k, s))


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        print(f"wrote {stem.with_suffix('.' + ext)}")
    plt.close(fig)


def plane(results: dict, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    n = 400
    grid = np.linspace(-4, 104, n)
    x, y = np.meshgrid(grid, grid)
    img = np.zeros((n, n, 4))
    img[..., :3] = to_rgb(ARMS["bridge"][1])
    img[..., 3] = 0.26 * np.clip((y - x) / 100.0, 0, 1) ** 1.25
    ax.imshow(img, origin="lower", extent=(-4, 104, -4, 104), zorder=0.5,
              interpolation="bilinear", aspect="auto")
    ax.plot([0, 100], [0, 100], ls="--", lw=0.9, color=RULE, zorder=1)
    ax.text(1, 102, "ideal", fontsize=7.4, ha="left", va="top",
            color=shade(ARMS["bridge"][1], 0.55), zorder=6)
    ax.text(86, 103, "defers to\neverything", fontsize=6.9, ha="right", va="top",
            color=INK_SOFT, zorder=6)
    ax.text(17, 0.0, "answers everything", fontsize=6.9, ha="left", va="bottom",
            color=INK_SOFT, zorder=6)
    ax.text(64, 61.5, "indiscriminate deferral", fontsize=7, rotation=45,
            rotation_mode="anchor", ha="center", va="top", color=INK_SOFT, zorder=2)

    best = None
    for bb, _, marker in BACKBONES:
        points = {arm: results["main"][arm].get(bb) for arm in results["main"]}
        points["prompted"] = results["instructed"].get(bb)
        for arm, p in points.items():
            if not p:
                continue
            colour = ARMS[arm][1]
            ax.scatter(p["SDR"], p["VRR"], s=52, marker=marker, zorder=5,
                       facecolors="none" if arm == "prompted" else colour,
                       edgecolors=shade(colour), linewidths=1.1)
        b = points.get("bridge")
        if b and (best is None or b["DD"] > best["DD"]):
            best = b
    if best:
        ink = shade(ARMS["bridge"][1], 0.55)
        ax.annotate("", xy=(best["SDR"], best["VRR"] - 1.8),
                    xytext=(best["SDR"], best["SDR"] + 1.0),
                    arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.38",
                                    lw=0.9, color=ink, shrinkA=0, shrinkB=0), zorder=6)
        ax.text(best["SDR"] + 1.8, (best["SDR"] + best["VRR"]) / 2,
                f"VRR - SDR\n{best['DD']:.1f} pp", fontsize=7, ha="left",
                va="center", color=ink, linespacing=1.3, zorder=6)

    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4, color=GRID, zorder=0.2)
    ax.set_axisbelow(True)
    ax.set_xlabel("SDR: deferral on stable facts (%)")
    ax.set_ylabel("VRR: deferral on volatile facts (%)")

    order = ["zero_shot", "prompted", "sft", "ca_avmd", "presence_only", "bridge"]
    arm_h = [mlines.Line2D([], [], ls="none", marker="o", markersize=7,
                           markerfacecolor="none" if k == "prompted" else ARMS[k][1],
                           markeredgecolor=shade(ARMS[k][1]), label=ARMS[k][0])
             for k in order]
    bb_h = [mlines.Line2D([], [], ls="none", marker=m, markersize=7,
                          markerfacecolor="none", markeredgecolor=INK_SOFT, label=name)
            for _, name, m in BACKBONES]
    # three legend columns: arms 1-3, arms 4-6, backbones
    fig.legend(handles=arm_h + bb_h, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.13), handletextpad=0.5,
               columnspacing=1.2)
    fig.tight_layout()
    save(fig, stem)


def exchange_rate(results: dict, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.16, 4.69))
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    ticks, labels = [], []
    for i, (bb, name, _) in enumerate(BACKBONES):
        base = results["main"]["ca_avmd"].get(bb)
        if not base:
            continue
        for j, arm in enumerate(("presence_only", "bridge")):
            p = results["main"][arm].get(bb)
            if not p:
                continue
            x = i * 1.2 + (j - 0.5) * 0.58
            d_sr, d_dd = p["SR"] - base["SR"], p["DD"] - base["DD"]
            colour = ARMS[arm][1]
            for value, hatch in ((d_sr, "////"), (d_dd, "")):
                ax.bar(x, value, 0.26, color=colour, edgecolor=shade(colour),
                       hatch=hatch, linewidth=0.8, zorder=2)
            ax.text(x, d_sr - 1.2, f"{d_sr:+.1f}", ha="center", va="top",
                    fontsize=7.5, color=INK, path_effects=halo, zorder=4)
            # a negative DD bar overlaps the SR bar, so its label sits above zero
            ax.text(x, max(d_dd, 0) + 1.2, f"{d_dd:+.1f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=INK, path_effects=halo, zorder=4)
            ticks.append(x)
            labels.append("presence\ngate" if arm == "presence_only" else "BRIDGe")
        ax.annotate(name, xy=(i * 1.2, -0.19), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=9, annotation_clip=False)
    ax.axhline(0, color=shade(GRID), lw=1.0, zorder=3)
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel(r"$\Delta$ vs CA-AVMD (pp)")
    ax.set_ylim(-36, 22)
    ax.set_yticks([-30, -20, -10, 0, 10, 20])
    ax.grid(True, axis="y", lw=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    fig.legend(handles=[
        mpatches.Patch(facecolor="white", edgecolor=INK, linewidth=0.8, hatch="////",
                       label=r"$\Delta$SR"),
        mpatches.Patch(facecolor="white", edgecolor=INK, linewidth=0.8,
                       label=r"$\Delta$(VRR$-$SDR)"),
    ], loc="lower center", ncol=2, frameon=False, handlelength=1.6, columnspacing=2.4,
        bbox_to_anchor=(0.5, 0.035))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("outputs/tables/results.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = ap.parse_args()
    results = json.loads(args.results.read_text())
    plane(results, args.out_dir / "fig_discrimination_plane")
    exchange_rate(results, args.out_dir / "fig_exchange_rate")


if __name__ == "__main__":
    main()
