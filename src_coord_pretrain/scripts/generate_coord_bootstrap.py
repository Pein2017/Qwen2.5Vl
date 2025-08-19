#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Tuple


def _resolve_path(p: str) -> str:
    """Convert relative paths to absolute paths relative to current working directory."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = pp.resolve()
    return str(pp)


def _gen_identity(
    max_coord: int, n: int, rng: random.Random, canonical_only: bool = False
) -> Iterable[Tuple[str, str, int]]:
    if canonical_only:
        # Canonical prompts for Phase A (reduced variance)
        prompts = [
            "What is `{n}` in coordinate space?",
            "Convert {n} to coordinate token.",
        ]
    else:
        # Full prompt variety for Phase B
        prompts = [
            "What is `{n}` in coordinate space?",
            "Convert {n} to coordinate token.",
            "In coord space, what is {n}?",
            "Please answer as coordinate token: {n}",
            "Transform {n} into coordinate format.",
            "Express {n} as a coordinate token.",
        ]

    # Edge emphasis: include more samples at boundaries
    edge_values = [0, 1, max_coord - 1, max_coord]
    edge_samples = min(n // 10, len(edge_values) * 5)  # 10% edge samples
    regular_samples = n - edge_samples

    # Generate edge samples
    for _ in range(edge_samples):
        v = rng.choice(edge_values)
        p = rng.choice(prompts).format(n=v)
        y = f"<|coord_{v}|>"
        yield p, y, v

    # Generate regular samples
    for _ in range(regular_samples):
        v = rng.randint(0, max_coord)
        p = rng.choice(prompts).format(n=v)
        y = f"<|coord_{v}|>"
        yield p, y, v


def _safe_div(a: int, b: int) -> int | None:
    if b == 0:
        return None  # sentinel
    return a // b


def _gen_arith(
    max_coord: int, n: int, rng: random.Random
) -> Iterable[Tuple[str, str, int]]:
    ops = ["+", "-", "*", "/"]
    for _ in range(n):
        op = rng.choice(ops)
        if op == "+":
            # a + b in range
            res = rng.randint(0, max_coord)
            a = rng.randint(0, res)
            b = res - a
        elif op == "-":
            # a - b in range
            a = rng.randint(0, max_coord)
            b = rng.randint(0, a)
            res = a - b
        elif op == "*":
            # a * b in range
            # sample res then factorize approximately
            res = rng.randint(0, max_coord)
            if res == 0:
                a, b = 0, rng.randint(0, max_coord)
            else:
                # pick a divisor of res (fallback to 1)
                divs = [d for d in range(1, int(math.sqrt(res)) + 1) if res % d == 0]
                a = rng.choice(divs)
                b = res // a
        elif op == "/":
            # a // b in range
            b = rng.randint(1, max(1, min(32, max_coord)))  # avoid huge divisors
            res = rng.randint(0, max_coord)
            a = res * b
        else:
            raise AssertionError

        if not (0 <= res <= max_coord):
            continue
        q = f"what is {a}{op}{b} in coordinate space?"
        y = f"<|coord_{res}|>"
        yield q, y, res


def _gen_reverse_mapping(
    max_coord: int, n: int, rng: random.Random
) -> Iterable[Tuple[str, str, int]]:
    """Generate reverse mapping samples: <|coord_N|> → "N" """
    prompts = [
        "What is <|coord_{n}|> as text?",
        "Convert <|coord_{n}|> to number",
        "What number does <|coord_{n}|> represent?",
        "Please convert <|coord_{n}|> to its numeric value",
    ]
    for _ in range(n):
        v = rng.randint(0, max_coord)
        p = rng.choice(prompts).format(n=v)
        y = str(v)  # Raw number as string
        yield p, y, v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coord bootstrap JSONL (text-only)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to JSONL output (absolute or relative)",
    )
    # New ratio-based API
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Total number of samples across all tasks (ratio mode)",
    )
    parser.add_argument(
        "--ratio_identity",
        type=float,
        default=None,
        help="Proportion for identity samples in [0,1]; 0 or None disables",
    )
    parser.add_argument(
        "--ratio_arithmetic",
        type=float,
        default=None,
        help="Proportion for arithmetic samples in [0,1]; 0 or None disables",
    )
    parser.add_argument(
        "--ratio_reverse",
        type=float,
        default=None,
        help="Proportion for reverse mapping samples in [0,1]; 0 or None disables",
    )
    parser.add_argument(
        "--canonical_prompts",
        action="store_true",
        help="Use canonical prompts only (reduced variance for Phase A)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable exact deduplication of identical (user, assistant) pairs before writing",
    )
    parser.add_argument("--max_coord", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out_path = _resolve_path(args.output)
    max_coord = int(args.max_coord)
    if max_coord <= 0:
        raise ValueError(f"max_coord must be positive, got {max_coord}")

    rng = random.Random(args.seed)

    # Validate ratios and allocate counts using largest-remainder method
    ratio_entries = [
        ("identity", args.ratio_identity),
        ("arithmetic", args.ratio_arithmetic),
        ("reverse_mapping", args.ratio_reverse),
    ]
    active = [(name, r) for name, r in ratio_entries if (r is not None and r > 0.0)]
    if not active:
        raise ValueError(
            "At least one of --ratio_identity/--ratio_arithmetic/--ratio_reverse must be > 0"
        )
    for name, r in active:
        if not (r >= 0.0):
            raise ValueError(f"Ratio {name} must be >= 0, got {r}")

    sum_ratios = sum(r for _, r in active)
    if sum_ratios <= 0.0:
        raise ValueError("Sum of active ratios must be > 0")

    num_samples = int(args.num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    # Normalize ratios
    normalized = [(name, r / sum_ratios) for name, r in active]
    # Base allocations and remainders
    allocations = []
    total_base = 0
    for name, r in normalized:
        exact = num_samples * r
        alloc = int(exact)
        rem = exact - alloc
        allocations.append((name, alloc, rem))
        total_base += alloc
    remaining = num_samples - total_base
    # Distribute remaining by largest remainder, tie-broken by stable name order
    allocations.sort(key=lambda x: (-x[2], x[0]))
    for i in range(remaining):
        name, alloc, rem = allocations[i]
        allocations[i] = (name, alloc + 1, rem)
    # Restore name order for readability
    allocations.sort(key=lambda x: x[0])
    counts = {name: alloc for name, alloc, _ in allocations}

    identity_count = counts.get("identity", 0)
    arith_count = counts.get("arithmetic", 0)
    reverse_count = counts.get("reverse_mapping", 0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect generated examples before optional deduplication
    examples = []

    # Generate identity samples
    if identity_count > 0:
        for q, y, res in _gen_identity(
            max_coord, identity_count, rng, canonical_only=args.canonical_prompts
        ):
            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": y},
                ],
                "meta": {"task": "identity", "result": res},
            }
            examples.append(obj)

    # Generate arithmetic samples
    if arith_count > 0:
        for q, y, res in _gen_arith(max_coord, arith_count, rng):
            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": y},
                ],
                "meta": {"task": "arithmetic", "result": res},
            }
            examples.append(obj)

    # Generate reverse mapping samples
    if reverse_count > 0:
        for q, y, res in _gen_reverse_mapping(max_coord, reverse_count, rng):
            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": y},
                ],
                "meta": {"task": "reverse_mapping", "result": res},
            }
            examples.append(obj)

    raw_count = len(examples)

    # Optional exact deduplication by (user, assistant) content pair
    if args.dedup:
        seen_pairs = set()
        unique_examples = []
        for ex in examples:
            user_text = ex["messages"][0]["content"]
            assistant_text = ex["messages"][1]["content"]
            key = (user_text, assistant_text)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unique_examples.append(ex)
        examples = unique_examples

    final_count = len(examples)

    # Calculate actual distribution after deduplication
    actual_counts = {"identity": 0, "arithmetic": 0, "reverse_mapping": 0}
    for ex in examples:
        task = ex["meta"]["task"]
        actual_counts[task] += 1

    # Write out
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Logs
    if args.dedup:
        removed = raw_count - final_count
        pct = (removed / raw_count * 100.0) if raw_count > 0 else 0.0
        print(f"Before dedup: {raw_count}")
        print(f"After dedup:  {final_count} (removed {removed}, {pct:.2f}%)")
    print(f"Wrote {final_count} samples to {out_path}")
    print(
        f"  - Requested total: {num_samples}; ratios (raw): "
        f"identity={args.ratio_identity}, arithmetic={args.ratio_arithmetic}, reverse={args.ratio_reverse}"
    )
    print(
        f"  - Allocations: identity={identity_count}, arithmetic={arith_count}, reverse={reverse_count}"
    )

    # Print actual distribution after deduplication
    if args.dedup and final_count > 0:
        actual_identity_ratio = actual_counts["identity"] / final_count
        actual_arithmetic_ratio = actual_counts["arithmetic"] / final_count
        actual_reverse_ratio = actual_counts["reverse_mapping"] / final_count
        print(f"  - Actual distribution after dedup:")
        print(
            f"    identity={actual_counts['identity']} ({actual_identity_ratio:.3f}), "
            f"arithmetic={actual_counts['arithmetic']} ({actual_arithmetic_ratio:.3f}), "
            f"reverse={actual_counts['reverse_mapping']} ({actual_reverse_ratio:.3f})"
        )


if __name__ == "__main__":
    main()
