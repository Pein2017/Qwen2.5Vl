#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Ensure project root is importable when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src_new.sampler.random_bucket import RandomBucketSampler  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    return samples


def compute_bucket_assignments(samples: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[int]]]:
    major_types: List[str] = []
    bucket_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, sample in enumerate(samples):
        mt = RandomBucketSampler._major_type_of(sample)
        major_types.append(mt)
        bucket_to_indices[mt].append(idx)

    return major_types, bucket_to_indices


def extract_types_from_sample(sample: Dict[str, Any]) -> Set[str]:
    """Extract object-type buckets present in a sample from objects[*].desc prefixes.

    Uses the sampler's prefix→bucket mapping to ensure alignment.
    """
    types: Set[str] = set()
    objs = sample.get("objects", []) or []
    for obj in objs:
        desc = obj.get("desc", "")
        if not isinstance(desc, str) or not desc:
            continue
        prefix = desc.split("/")[0].strip()
        bucket = RandomBucketSampler._map_prefix_to_bucket(prefix)
        if bucket is not None:
            types.add(bucket)
    if not types:
        types.add("misc")
    return types


def build_type_index(samples: List[Dict[str, Any]]) -> Tuple[List[Set[str]], Dict[str, List[int]]]:
    """Build per-sample type sets and an index mapping type→list of sample indices."""
    per_sample_types: List[Set[str]] = []
    type_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, sample in enumerate(samples):
        tset = extract_types_from_sample(sample)
        per_sample_types.append(tset)
        for t in tset:
            type_to_indices[t].append(idx)

    return per_sample_types, type_to_indices


def compute_overlap_group_sizes(per_sample_types: List[Set[str]], type_to_indices: Dict[str, List[int]]) -> List[int]:
    """For each sample, compute the size of the union of all samples sharing any type with it (excluding itself)."""
    sizes: List[int] = []
    n = len(per_sample_types)
    for idx in range(n):
        tset = per_sample_types[idx]
        overlap_indices: Set[int] = set()
        for t in tset:
            overlap_indices.update(type_to_indices.get(t, []))
        if idx in overlap_indices:
            overlap_indices.remove(idx)
        sizes.append(len(overlap_indices))
    return sizes


def write_per_sample_csv_major(out_path: Path, major_types: List[str], bucket_to_indices: Dict[str, List[int]]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("sample_index,bucket,bucket_size\n")
        for idx, mt in enumerate(major_types):
            size = len(bucket_to_indices.get(mt, []))
            f.write(f"{idx},{mt},{size}\n")


def write_per_sample_csv_overlap(out_path: Path, per_sample_types: List[Set[str]], overlap_sizes: List[int]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("sample_index,types,overlap_size\n")
        for idx, tset in enumerate(per_sample_types):
            types_str = ";".join(sorted(tset))
            f.write(f"{idx},{types_str},{overlap_sizes[idx]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect grouping sizes per sample using sampler mapping")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_PROJECT_ROOT / "data/ds_v2_full/train.jsonl"),
        help="Path to input JSONL (default: data/ds_v2_full/train.jsonl)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(_PROJECT_ROOT / "bucket_sizes_train.csv"),
        help="Path to write per-sample CSV (default: bucket_sizes_train.csv at project root)",
    )
    parser.add_argument(
        "--show_top",
        type=int,
        default=0,
        help="Print first N sample rows from the CSV to stdout (default: 0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overlap", "major"],
        default="overlap",
        help="Grouping mode: 'overlap' (any shared object type) or 'major' (single dominant type)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output_csv).resolve()

    print(f"Reading samples from: {input_path}")
    samples = read_jsonl(input_path)
    total = len(samples)
    print(f"Loaded {total} samples")

    if args.mode == "major":
        major_types, bucket_to_indices = compute_bucket_assignments(samples)
        counts = Counter(major_types)
        print("\nMajor-type bucket counts (sorted):")
        for bucket, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = (cnt / total * 100.0) if total > 0 else 0.0
            print(f"  {bucket:12s} : {cnt:6d}  ({pct:5.1f}%)")
        write_per_sample_csv_major(output_path, major_types, bucket_to_indices)
    else:
        per_sample_types, type_to_indices = build_type_index(samples)
        overlap_sizes = compute_overlap_group_sizes(per_sample_types, type_to_indices)

        # Summaries
        counts = Counter()
        for tset in per_sample_types:
            for t in tset:
                counts[t] += 1

        print("\nObject-type presence counts across samples (sorted):")
        for t, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = (cnt / total * 100.0) if total > 0 else 0.0
            print(f"  {t:12s} : {cnt:6d}  ({pct:5.1f}%)")

        print("\nOverlap group size stats:")
        if overlap_sizes:
            avg = sum(overlap_sizes) / len(overlap_sizes)
            print(f"  mean: {avg:.2f}")
            for q in (0, 10, 25, 50, 75, 90, 95, 99, 100):
                k = max(0, min(len(overlap_sizes) - 1, int(round(q / 100.0 * (len(overlap_sizes) - 1)))))
                val = sorted(overlap_sizes)[k]
                print(f"  p{q:02d}: {val}")
        else:
            print("  no data")

        write_per_sample_csv_overlap(output_path, per_sample_types, overlap_sizes)

    print(f"\nWrote per-sample grouping sizes to: {output_path}")

    # Optionally print the first N rows (skip header)
    if args.show_top and args.show_top > 0:
        print(f"\nFirst {args.show_top} rows:")
        with output_path.open("r", encoding="utf-8") as f:
            next(f)  # header
            for i, line in enumerate(f):
                if i >= args.show_top:
                    break
                print(line.rstrip())


if __name__ == "__main__":
    main()
