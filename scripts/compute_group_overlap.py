#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_DIR_NAMES = ("审核通过", "审核不通过")


@dataclass(frozen=True)
class IndexedImage:
    sample_index: int
    source_path: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_filename(p: str) -> str:
    return os.path.basename(p).lower()


def load_all_samples_index(jsonl_path: Path) -> Dict[str, IndexedImage]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL not found: {jsonl_path}. Expected the training all-samples index file."
        )

    index: Dict[str, IndexedImage] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON at line {i+1} in {jsonl_path}: {e}. "
                    f"Fix the malformed line or regenerate the file."
                )
            images = rec.get("images") or []
            if not isinstance(images, list):
                raise ValueError(
                    f"Record {i} has non-list 'images' field (type={type(images)})."
                )
            for p in images:
                if not isinstance(p, str):
                    continue
                fname = normalize_filename(p)
                # Keep the first occurrence; presence is what matters for overlap.
                if fname not in index:
                    index[fname] = IndexedImage(sample_index=i, source_path=p)
    if not index:
        raise RuntimeError(
            f"No images indexed from {jsonl_path}. Ensure the file has 'images' fields with paths."
        )
    return index


def iter_label_dirs(group_root: Path) -> List[Path]:
    out: List[Path] = []
    if not group_root.exists():
        return out
    # Include direct children pattern and recursive pattern for robustness
    for label_name in LABEL_DIR_NAMES:
        direct = group_root / label_name
        if direct.is_dir():
            out.append(direct)
    for found in group_root.glob("**/*"):
        if found.is_dir() and found.name in LABEL_DIR_NAMES and found not in out:
            out.append(found)
    return out


def iter_group_dirs(group_root: Path) -> List[Path]:
    out: List[Path] = []
    for label_dir in iter_label_dirs(group_root):
        for child in label_dir.iterdir():
            if child.is_dir():
                out.append(child)
    return out


def extract_mission_and_label(group_dir: Path) -> Tuple[str, str]:
    parts = group_dir.parts
    for idx, name in enumerate(parts):
        if name in LABEL_DIR_NAMES:
            mission = parts[idx - 1] if idx - 1 >= 0 else ""
            return mission, name
    return "", ""


def list_group_images(group_dir: Path) -> List[str]:
    images: List[str] = []
    try:
        for p in group_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
                images.append(p.name)
    except Exception as e:
        raise RuntimeError(f"Failed to list images under group dir: {group_dir}: {e}")
    return images


def should_include_label(label: str, label_filter: str) -> bool:
    if label_filter == "all":
        return True
    if label_filter == "pass":
        return label == "审核通过"
    if label_filter == "fail":
        return label == "审核不通过"
    raise ValueError(
        f"Invalid label_filter={label_filter}. Expected one of: all|pass|fail"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-group image overlap ratios against all_samples.jsonl and sort descending."
        )
    )
    parser.add_argument(
        "--all_samples",
        type=str,
        default="/data3/Qwen2.5-VL-main/data/ds_v2_full/all_samples.jsonl",
        help="Path to all_samples.jsonl",
    )
    parser.add_argument(
        "--group_root",
        type=str,
        default="/data3/Qwen2.5-VL-main/group_data/bbu_scene_2.0_order",
        help="Root directory containing mission subdirectories with 审核通过/审核不通过",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data3/Qwen2.5-VL-main/data/cleaning_reports/group_overlap_sorted.jsonl",
        help="Where to write the sorted JSONL report",
    )
    parser.add_argument(
        "--label_filter",
        type=str,
        default="all",
        choices=["all", "pass", "fail"],
        help="Which label dirs to include in the scan",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Print top-K groups to stdout for quick inspection",
    )

    args = parser.parse_args()

    all_samples_path = Path(args.all_samples)
    group_root = Path(args.group_root)
    output_path = Path(args.output)

    if not group_root.exists():
        raise FileNotFoundError(
            f"group_root does not exist: {group_root}. Provide a valid --group_root."
        )

    ensure_dir(output_path.parent)

    index = load_all_samples_index(all_samples_path)

    # Collect records
    records: List[Dict[str, object]] = []
    num_groups_scanned = 0

    for gdir in sorted(iter_group_dirs(group_root)):
        mission, label_name = extract_mission_and_label(gdir)
        if not should_include_label(label_name, args.label_filter):
            continue
        images = list_group_images(gdir)
        if not images:
            continue
        num_groups_scanned += 1

        matched: List[str] = []
        unmatched: List[str] = []
        for fname in images:
            key = fname.lower()
            if key in index:
                matched.append(fname)
            else:
                unmatched.append(fname)

        overlap = len(matched)
        total = len(images)
        ratio = overlap / total if total > 0 else 0.0

        rec = {
            "group_dir": str(gdir),
            "mission": mission,
            "label_dir": label_name,
            "num_images": total,
            "overlap_images": overlap,
            "overlap_ratio": round(ratio, 6),
            "matched_images": matched,
            "unmatched_images": unmatched,
        }
        records.append(rec)

    # Sort by overlap_ratio desc, then by overlap_images desc, then by num_images desc, then by path
    records.sort(
        key=lambda r: (
            float(r["overlap_ratio"]),
            int(r["overlap_images"]),
            int(r["num_images"]),
            r["group_dir"],
        ),
        reverse=True,
    )

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as writer:
        for rec in records:
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print concise top-K summary
    top_k = max(0, int(args.top_k))
    print(json.dumps({
        "total_groups": len(records),
        "groups_scanned": num_groups_scanned,
        "label_filter": args.label_filter,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))

    if top_k > 0:
        print("\nTop groups by overlap ratio:")
        for i, rec in enumerate(records[:top_k], start=1):
            print(
                f"{i:>3}. ratio={rec['overlap_ratio']:.6f} "
                f"({rec['overlap_images']}/{rec['num_images']}) "
                f"{rec['label_dir']} | {rec['mission']} | {rec['group_dir']}"
            )


if __name__ == "__main__":
    main()
