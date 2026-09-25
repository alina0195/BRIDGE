#!/usr/bin/env python3
"""Download EvoWiki and build the entity-level train / eval splits.

    python scripts/prepare_data.py --output-dir data/evowiki
    python scripts/prepare_data.py --output-dir data/evowiki --skip-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.data import build_splits, download_evowiki, load_evowiki  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="Raw EvoWiki JSON files (default: <output-dir>/raw).")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = args.raw_dir or args.output_dir / "raw"
    if not args.skip_download:
        raw_dir = download_evowiki(args.output_dir)
    counts = build_splits(load_evowiki(raw_dir), args.output_dir,
                          train_ratio=args.train_ratio, seed=args.seed)
    for name, n in counts.items():
        print(f"  {name:14s} {n:6d}")


if __name__ == "__main__":
    main()
