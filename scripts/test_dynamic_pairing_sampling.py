#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only dynamic pairing sampling test (JSON pipeline)

This script mimics the dynamic contrastive pairing process without initializing any model or GPU.
It focuses on the data sampling logic:
  - Build synthetic samples with object-type buckets from desc prefixes
  - Simulate per-sample hardness EMA and normalization to weights
  - Construct candidate pools per student (bucket union)
  - Compare teacher selection distributions across temperatures

Run:
  python scripts/test_dynamic_pairing_sampling.py --num_samples 200 --epochs 50 \
      --teacher_ratio 0.6 --pool_fraction 0.4 --pool_max 4096 \
      --tau_list 0.5 1.2

Expected output:
  - Top-k teachers by selection frequency for each temperature
  - Correlation between selection freq and hardness weights
  - Simple coverage and histogram stats
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src_new_json.sampling.bucketed_sampling import (
    BucketedSamplingEngine,
    SamplingConfig,
)

# ---- Synthetic data builders ----

BUCKET_PREFIXES = [
    ("BBU", "bbu"),
    ("Shield", "bbu_shield"),
    ("ConnectPoint", "connect_point"),
    ("Fiber", "fiber"),
    ("Wire", "wire"),
    ("Label", "label"),
]


def build_synthetic_sample(prefix_key: str, idx: int) -> Dict:
    """Build one synthetic sample dict with a desc prefix mapping to a bucket.

    Only desc is used by the sampler; geometry is ignored by BucketedSamplingEngine.
    """
    prefix = prefix_key
    desc = f"{prefix}/synthetic_attr_{idx}"
    # Minimal object list: one object with desc
    return {
        "id": f"s_{idx}",
        "images": [f"/tmp/placeholder_{idx}.jpg"],
        "objects": [
            {
                "desc": desc,
                # Geometry keys are irrelevant for the sampler; included for completeness
                "bbox_2d": [0, 0, 10, 10],
            }
        ],
        "width": 64,
        "height": 64,
    }


def build_synthetic_samples(num_samples: int, rng: random.Random) -> List[Dict]:
    samples: List[Dict] = []
    # Assign each sample 1-2 prefixes with some structure to encourage overlap
    for i in range(num_samples):
        # Bias buckets to create non-uniform overlaps
        base_idx = int(abs(rng.gauss(mu=len(BUCKET_PREFIXES) / 2, sigma=1.5))) % len(
            BUCKET_PREFIXES
        )
        prefix_primary = BUCKET_PREFIXES[base_idx][0]
        # Small prob to add a secondary type
        prefixes = [prefix_primary]
        if rng.random() < 0.35:
            secondary = BUCKET_PREFIXES[rng.randrange(len(BUCKET_PREFIXES))][0]
            if secondary != prefix_primary:
                prefixes.append(secondary)
        # For simplicity, attach only the first prefix to the desc; pool builder considers unions
        samples.append(build_synthetic_sample(prefixes[0], i))
    return samples


# ---- Hardness (EMA) simulation and normalization ----

def simulate_hardness_ema(num_samples: int, rng: random.Random) -> List[float]:
    """Create a synthetic hardness EMA vector in [0,1] with non-uniform distribution.

    We generate a base difficulty per sample and then smooth it slightly to emulate EMA.
    """
    base = np.clip(np.array([rng.random() for _ in range(num_samples)]), 0.0, 1.0)
    # Add a small trend so some indices are systematically harder
    trend = np.linspace(0.0, 0.4, num_samples)
    ema = np.clip(0.6 * base + 0.4 * trend, 0.0, 1.0)
    return ema.tolist()


def normalize_hardness_to_weights(ema: List[float]) -> List[float]:
    """Replicate dataset normalization: clip by percentiles (p10..p90) then clamp to [0,1]."""
    arr = np.array(ema, dtype=np.float32)
    if not np.isfinite(arr).any():
        return [0.0 for _ in ema]
    p10 = np.quantile(arr, 0.10)
    p90 = np.quantile(arr, 0.90)
    denom = max(1e-6, float(p90 - p10))
    norm = np.clip((arr - p10) / denom, 0.0, 1.0)
    return norm.tolist()


# ---- Sampling experiment ----

def run_sampling(
    samples: List[Dict],
    weights: List[float] | None,
    tau: float,
    teacher_ratio: float,
    pool_fraction: float,
    pool_max: int,
    base_seed: int,
    epochs: int,
) -> Dict[str, any]:
    engine = BucketedSamplingEngine()
    cfg = SamplingConfig(
        temperature=float(max(1e-6, tau)),
        target_assignment="current",
        cross_bucket_explore_prob=0.0,
        pool_fraction=float(pool_fraction),
        pool_max=int(pool_max),
    )

    n = len(samples)
    total_pairs = 0
    context_counts: Counter[int] = Counter()
    per_student: Dict[int, Counter[int]] = {i: Counter() for i in range(n)}

    for e in range(epochs):
        episode_map, _metrics = engine.build_epoch_map(
            samples=samples,
            cfg=cfg,
            base_seed=base_seed,
            epoch_idx=e,
            teacher_ratio=float(teacher_ratio),
            is_eval=False,
            sample_weights=weights,
            rank=0,
            worker_id=0,
        )
        for tgt, spec in episode_map.items():
            j = spec.context_idx
            if j is None:
                continue
            context_counts[int(j)] += 1
            per_student[int(tgt)][int(j)] += 1
            total_pairs += 1

    # Compute frequency vector aligned with weights
    freq = np.zeros(n, dtype=np.int64)
    for j, c in context_counts.items():
        if 0 <= int(j) < n:
            freq[int(j)] = int(c)

    return {
        "total_pairs": total_pairs,
        "context_hist": context_counts,
        "per_student": per_student,
        "freq": freq,
    }


def summarize(freq: np.ndarray, weights: List[float] | None, top_k: int = 10) -> str:
    n = len(freq)
    idxs = np.argsort(-freq)[: min(top_k, n)]
    lines = []
    for r, j in enumerate(idxs):
        w = None if weights is None else float(weights[j])
        lines.append(f"#{r+1}: teacher={j:4d} freq={int(freq[j])} weight={w if w is not None else 'NA'}")
    # Correlation
    corr = None
    if weights is not None and np.any(freq > 0):
        try:
            corr = float(np.corrcoef(np.array(weights, dtype=np.float32), freq.astype(np.float32))[0, 1])
        except Exception:
            corr = None
    return "\n".join(lines + [f"pearson_corr(freq, weight)={corr}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--teacher_ratio", type=float, default=0.6)
    ap.add_argument("--pool_fraction", type=float, default=0.4)
    ap.add_argument("--pool_max", type=int, default=4096)
    ap.add_argument("--tau_list", type=float, nargs="+", default=[0.5, 1.2])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    samples = build_synthetic_samples(int(args.num_samples), rng)

    # Simulate hardness EMA and normalize to weights in [0,1]
    ema = simulate_hardness_ema(len(samples), rng)
    weights = normalize_hardness_to_weights(ema)

    print(f"Synthetic dataset: N={len(samples)}, epochs={args.epochs}")
    print(f"Teacher ratio={args.teacher_ratio}, pool_fraction={args.pool_fraction}, pool_max={args.pool_max}")

    # Baseline (uniform) for reference
    res_uniform = run_sampling(
        samples=samples,
        weights=None,  # uniform
        tau=1.0,
        teacher_ratio=args.teacher_ratio,
        pool_fraction=args.pool_fraction,
        pool_max=args.pool_max,
        base_seed=args.seed,
        epochs=args.epochs,
    )
    print("\n=== Uniform (no hardness, tau=1.0) ===")
    print(f"total_pairs={res_uniform['total_pairs']}")
    print(summarize(res_uniform["freq"], None))

    # Loss-aware runs with different temperatures
    for tau in args.tau_list:
        res = run_sampling(
            samples=samples,
            weights=weights,
            tau=float(tau),
            teacher_ratio=args.teacher_ratio,
            pool_fraction=args.pool_fraction,
            pool_max=args.pool_max,
            base_seed=args.seed,
            epochs=args.epochs,
        )
        print(f"\n=== Loss-aware (tau={tau}) ===")
        print(f"total_pairs={res['total_pairs']}")
        print(summarize(res["freq"], weights))

    print("\nNote: A lower tau should concentrate mass on high-weight teachers (higher selection freq, higher corr).\n"
          "If results look identical across taus, enlarge pool_fraction/pool_max or increase epochs.")


if __name__ == "__main__":
    main()
