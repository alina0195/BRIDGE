#!/usr/bin/env python3
"""Cosine similarity between the volatility-probe directions of zero-shot
and the five fine-tuned variants of one backbone (prompt_last position,
deepest probed layer).

    python scripts/plots/plot_probe_cosine.py --backbones llama
Reads  outputs/mechanistic/{bb}/probes/{variant}/activations/
       layer_<L>_prompt_last_direction.npy
Writes outputs/figures/volatility_cosine_{bb}.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS = [
    ("zero_shot", "Zero-Shot"),
    ("sft", "SFT"),
    ("vmd", "VMD"),
    ("avmd", "AVMD"),
    ("ca_vmd", "CA-VMD"),
    ("ca_avmd", "CA-AVMD"),
]
EXTRACTION = "prompt_last"


def load_directions(variant_dir: Path) -> dict[int, np.ndarray]:
    out = {}
    for f in (variant_dir / "activations").glob(f"layer_*_{EXTRACTION}_direction.npy"):
        out[int(f.name.split("_")[1])] = np.load(f).reshape(-1)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def plot(probe_root: Path, out_stem: Path, layer: int | None) -> None:
    dirs = {}
    for key, label in VARIANTS:
        dd = load_directions(probe_root / key)
        if dd:
            dirs[label] = dd[max(dd) if layer is None else layer]
    labels = list(dirs)
    n = len(labels)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = cosine(dirs[labels[i]], dirs[labels[j]])

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.6 else "#222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech-root", type=Path, default=Path("outputs/mechanistic"))
    ap.add_argument("--backbones", default="llama mistral qwen")
    ap.add_argument("--layer", type=int, default=None,
                    help="Absolute layer index; default is the deepest probed layer.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = ap.parse_args()
    for bb in args.backbones.split():
        out = args.out_dir / f"volatility_cosine_{bb}"
        plot(args.mech_root / bb / "probes", out, args.layer)
        print(f"wrote {out}.pdf and .png")


if __name__ == "__main__":
    main()
