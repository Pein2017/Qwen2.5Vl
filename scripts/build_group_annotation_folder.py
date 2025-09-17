#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

from data_conversion.summary_builder import build_summary_from_objects


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_DIR_NAMES = ("审核通过", "审核不通过")


@dataclass(frozen=True)
class ImageSummary:
    sample_index: int
    source_path: str
    object_descs: List[str]
    has_negative: bool
    negatives: List[str]
    objects: List[Dict[str, Any]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_filename(p: str) -> str:
    return os.path.basename(p).lower()


def load_all_samples_summaries(jsonl_path: Path) -> Dict[str, ImageSummary]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL not found: {jsonl_path}. Expected the training all-samples file."
        )

    index: Dict[str, ImageSummary] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON at line {i+1} in {jsonl_path}: {e}."
                )
            images = rec.get("images") or []
            if not isinstance(images, list):
                raise ValueError(
                    f"Record {i} has non-list 'images' (type={type(images)})."
                )
            meta = rec.get("meta") or {}
            has_negative = bool(meta.get("has_negative", False))
            neg_list = meta.get("negatives", [])
            if not isinstance(neg_list, list):
                neg_list = []
            obj_descs: List[str] = []
            for obj in rec.get("objects", []) or []:
                desc = obj.get("desc")
                if isinstance(desc, str) and desc.strip():
                    obj_descs.append(desc.strip())
            # keep raw objects for summary builder
            obj_list = rec.get("objects", []) or []
            if not isinstance(obj_list, list):
                obj_list = []
            for p in images:
                if not isinstance(p, str):
                    continue
                fname = normalize_filename(p)
                if fname not in index:
                    index[fname] = ImageSummary(
                        sample_index=i,
                        source_path=p,
                        object_descs=list(obj_descs),
                        has_negative=has_negative,
                        negatives=list(neg_list),
                        objects=list(obj_list),
                    )
    if not index:
        raise RuntimeError(
            f"No images indexed from {jsonl_path}. Ensure 'images' fields contain paths."
        )
    return index


def read_full_overlap_records(full_overlap_path: Path) -> List[Dict[str, object]]:
    if not full_overlap_path.exists():
        raise FileNotFoundError(
            f"full_overlap jsonl not found: {full_overlap_path}."
        )
    records: List[Dict[str, object]] = []
    with full_overlap_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON at line {i+1} in {full_overlap_path}: {e}."
                )
            # Expect required keys
            for key in ("group_dir", "mission", "label_dir", "num_images"):
                if key not in rec:
                    raise ValueError(
                        f"Missing '{key}' in record at line {i+1} of {full_overlap_path}."
                    )
            # Matched images preferred; may fallback to listing from dir later
            records.append(rec)
    if not records:
        raise RuntimeError(
            f"No records loaded from {full_overlap_path}."
        )
    return records


def list_group_images(group_dir: Path) -> List[str]:
    out: List[str] = []
    for p in group_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            out.append(p.name)
    return out


def build_group_annotation(
    full_overlap_path: Path,
    all_samples_path: Path,
    output_root: Path,
) -> Dict[str, int]:
    ensure_dir(output_root)
    index = load_all_samples_summaries(all_samples_path)
    records = read_full_overlap_records(full_overlap_path)

    processed_groups = 0
    copied_images = 0
    missing_in_index = 0

    for rec in records:
        group_dir = Path(rec["group_dir"])  # absolute path preferred
        mission = str(rec.get("mission", "")).strip()
        label_dir = str(rec.get("label_dir", "")).strip()
        if label_dir not in LABEL_DIR_NAMES:
            raise ValueError(
                f"Unexpected label_dir='{label_dir}' for group {group_dir}."
            )
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Group dir not found: {group_dir}")

        # Determine images to use
        matched_images = rec.get("matched_images")
        if isinstance(matched_images, list) and matched_images:
            images = [str(x) for x in matched_images]
        else:
            # Fallback: list directory
            images = list_group_images(group_dir)
        if not images:
            continue

        # Destination dir: group_annotation/<mission>/<label_dir>/<group_id>
        group_id = group_dir.name
        dest_dir = output_root / mission / label_dir / group_id
        ensure_dir(dest_dir)

        # Prepare summary writer
        summary_path = dest_dir / "summary.jsonl"
        with summary_path.open("w", encoding="utf-8") as writer:
            for fname in images:
                src_path = group_dir / fname
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"Missing image '{fname}' in {group_dir}"
                    )
                # Copy (avoid extended attributes to prevent OSError 524)
                dst_path = dest_dir / fname
                try:
                    shutil.copyfile(str(src_path), str(dst_path))
                    try:
                        shutil.copymode(str(src_path), str(dst_path))
                        shutil.copystat(str(src_path), str(dst_path))
                    except Exception:
                        # Best-effort for metadata; ignore failures
                        pass
                except Exception as e:
                    raise RuntimeError(f"Failed to copy {src_path} -> {dst_path}: {e}")
                copied_images += 1

                # Summarize from all_samples index
                key = fname.lower()
                info = index.get(key)
                if info is None:
                    missing_in_index += 1
                    raise KeyError(
                        f"Image '{fname}' from {group_dir} not found in all_samples index"
                    )

                desc_summary = build_summary_from_objects(info.objects)

                writer.write(
                    json.dumps(
                        {
                            "image": fname,
                            "group_id": group_id,
                            "group_dir": str(group_dir),
                            "mission": mission,
                            "label_dir": label_dir,
                            "sample_index": info.sample_index,
                            "all_samples_source_path": info.source_path,
                            "objects_count": len(info.objects),
                            "desc_summary": desc_summary,
                            "meta_has_negative": bool(info.has_negative),
                            "meta_negatives": info.negatives,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        processed_groups += 1

    return {
        "processed_groups": processed_groups,
        "copied_images": copied_images,
        "missing_in_index": missing_in_index,
        "output_root_exists": int(output_root.exists()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build group_annotation folder from full-overlap groups, copy images, and write per-group summary.jsonl."
        )
    )
    parser.add_argument(
        "--full_overlap",
        type=str,
        default="/data3/Qwen2.5-VL-main/data/cleaning_reports/full_overlap.jsonl",
        help="Path to the JSONL containing groups with overlap_ratio==1",
    )
    parser.add_argument(
        "--all_samples",
        type=str,
        default="/data3/Qwen2.5-VL-main/data/ds_v2_full/all_samples.jsonl",
        help="Path to all_samples.jsonl used for image summaries",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data3/Qwen2.5-VL-main/group_annotation",
        help="Output root directory to create the folder structure",
    )

    args = parser.parse_args()

    full_overlap_path = Path(args.full_overlap)
    all_samples_path = Path(args.all_samples)
    output_root = Path(args.output_root)

    stats = build_group_annotation(full_overlap_path, all_samples_path, output_root)
    print(json.dumps({**stats, "output_root": str(output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
