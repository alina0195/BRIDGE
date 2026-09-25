#!/usr/bin/env python3
"""Fine-tune a LoRA adapter with one of the five training objectives.

    PYTHONHASHSEED=42 python scripts/train.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct --objective ca_avmd \
        --data data/evowiki/train.jsonl --output-dir outputs/models/llama_ca_avmd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.training import OBJECTIVES, TrainingConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--objective", required=True, choices=OBJECTIVES)
    ap.add_argument("--data", type=Path, default=Path("data/evowiki/train.jsonl"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split-strategy", choices=("per_example", "entity"),
                    default="per_example",
                    help="How volatile examples are assigned to the grounded or bare "
                         "half (ca_vmd, ca_avmd): per example, or per entity.")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        objective=args.objective,
        split_strategy=args.split_strategy,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )
    train(config, args.data, args.output_dir)


if __name__ == "__main__":
    main()
