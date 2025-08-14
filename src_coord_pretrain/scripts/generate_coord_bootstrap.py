#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Tuple


def _abs_path(p: str) -> str:
    pp = Path(p)
    if not pp.is_absolute():
        # Convert relative path to absolute path
        pp = pp.resolve()
    return str(pp)


def _gen_identity(
    max_coord: int, n: int, rng: random.Random
) -> Iterable[Tuple[str, str, int]]:
    prompts = [
        "What is `{n}` in coordinate space?",
        "Convert {n} to coordinate token.",
        "In coord space, what is {n}?",
        "Please answer as coordinate token: {n}",
    ]
    for _ in range(n):
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
    parser.add_argument("--num_identity", type=int, required=True)
    parser.add_argument("--num_arithmetic", type=int, required=True)
    parser.add_argument("--max_coord", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out_path = _abs_path(args.output)
    max_coord = int(args.max_coord)
    if max_coord <= 0:
        raise ValueError(f"max_coord must be positive, got {max_coord}")

    rng = random.Random(args.seed)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q, y, res in _gen_identity(max_coord, args.num_identity, rng):
            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": y},
                ],
                "meta": {"task": "identity", "result": res},
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
        for q, y, res in _gen_arith(max_coord, args.num_arithmetic, rng):
            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": y},
                ],
                "meta": {"task": "arithmetic", "result": res},
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    main()
