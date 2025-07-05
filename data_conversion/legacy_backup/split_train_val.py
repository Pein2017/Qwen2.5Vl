#!/usr/bin/env python3
"""
Train/Val Splitter for Clean JSONL Format

Splits a clean-format JSONL file (with keys `images`, `objects`) into
separate train and validation JSONL files.

Example usage:
    python split_train_val.py \
        --input_jsonl data_conversion/student_combined.jsonl \
        --output_train data/chinese-train.jsonl \
        --output_val   data/chinese-val.jsonl \
        --val_ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def read_samples(path: Path) -> List[dict]:
    """Read all JSON lines from *path* into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_samples(samples: List[dict], path: Path) -> None:
    """Write *samples* to *path* in JSONL format with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for s in samples:
            fw.write(json.dumps(s, ensure_ascii=False) + "\n")


def split_train_val(
    samples: List[dict], val_ratio: float, seed: int
) -> tuple[List[dict], List[dict]]:
    """Shuffle *samples* and return (train, val) splits."""
    rng = random.Random(seed)
    rng.shuffle(samples)
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    return train_samples, val_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Split clean JSONL into train/val")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_val", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    samples = read_samples(in_path)
    if not samples:
        raise ValueError("No samples found in input JSONL")

    train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)

    write_samples(train_samples, Path(args.output_train))
    write_samples(val_samples, Path(args.output_val))

    print(
        f"✅ Split complete: {len(train_samples)} train, {len(val_samples)} val samples →"
        f" {args.output_train}, {args.output_val}"
    )


if __name__ == "__main__":
    main()
